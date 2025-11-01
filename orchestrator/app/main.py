from __future__ import annotations
# HARD BAN (permanent): Never add Pydantic, SQLAlchemy, or any ORM/CSV/Parquet libs to this service.
# - No Pydantic models. Use plain dicts + hand-rolled validators only.
# - No SQLAlchemy/ORM. Use asyncpg with pooling and raw SQL only.
# - No CSV/Parquet persistence. JSON/NDJSON only for manifests/logs.
#
# JSON POLICY (critical):
# - Never call json.loads directly in this service. All JSON coming from LLMs, tools, env/config,
#   or external services MUST be parsed via JSONParser().parse with an explicit expected structure.
# - Always pass an expected schema (dict/list with primitive types) to enforce shape and defaults;
#   avoid parser(..., {}) unless the value is truly open‑ended (e.g., opaque metadata).
# - When receiving strings that may be JSON (e.g., plan/workflow/metadata), coerce using JSONParser
#   BEFORE any .get or type‑dependent access to prevent 'str'.get type errors.
#
# Historical/rationale (orchestrator):
# - Planner/Executors (Qwen + GPT-OSS) are coordinated with explicit system guidance to avoid refusals.
# - Tools are selected SEMANTICALLY (not keywords). On refusal, we may force a single best-match tool
#   with minimal arguments, but we never overwrite model output; we only append a short Status.
# - JSON parsing uses a hardened JSONParser with expected structures to survive malformed LLM JSON.
# - All DB access is via asyncpg; JSONB writes use json.dumps and ::jsonb to avoid type issues.
# - RAG (pgvector) initializes extension + indexes at startup and uses a simple cache.
# - OpenAI-compatible /v1/chat/completions returns full Markdown: main answer first, then "### Tool Results"
#   (film_id, job_id(s), errors), then an "### Appendix — Model Answers" with raw Qwen/GPT-OSS responses (trimmed).
# - We never replace the main content with status; instead we append a Status block only when empty/refusal.
# - Long-running film pipeline: film_create → film_add_scene (ComfyUI jobs via /jobs) → film_compile (n8n or local assembler).
# - Keep LLM models warm: we set options.keep_alive=24h on every Ollama call to avoid reloading between requests.

import os
from types import SimpleNamespace
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
import time
import traceback

import httpx
import requests
import re
import asyncpg
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi.middleware.cors import CORSMiddleware
from .json_parser import JSONParser
from .determinism import seed_router as det_seed_router, seed_tool as det_seed_tool, round6 as det_round6
from .icw.windowed_solver import solve as windowed_solve
from .jsonio.normalize import normalize_to_envelope
from .jsonio.stitch import merge_envelopes as stitch_merge_envelopes, stitch_openai as stitch_openai_final
from .router.route import route_for_request
from .ablation.core import ablate as ablate_env
from .ablation.export import write_facts_jsonl as ablate_write_facts
from .code_loop.super_loop import run_super_loop
from .state.checkpoints import append_event as _append_event
from .state.lock import acquire_lock as _acquire_lock, release_lock as _release_lock
from .artifacts.manifest import add_manifest_row as _manifest_add_row, write_manifest_atomic as _manifest_write
from .state.ids import step_id as _step_id
from .research.orchestrator import run_research
from .jobs.state import get_job as _get_orcjob, request_cancel as _orcjob_cancel
from .artifacts.shard import open_shard as _art_open_shard, append_jsonl as _art_append_jsonl, _finalize_shard as _art_finalize
from .artifacts.shard import newest_part as _art_newest_part, list_parts as _art_list_parts
from .artifacts.manifest import add_manifest_row as _art_manifest_add, write_manifest_atomic as _art_manifest_write
async def _db_insert_run(trace_id: str, mode: str, seed: int, pack_hash: Optional[str], request_json: Dict[str, Any]) -> Optional[int]:
    try:
        pool = await get_pg_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO run(trace_id, mode, seed, pack_hash, request_json) VALUES($1,$2,$3,$4,$5) ON CONFLICT (trace_id) DO UPDATE SET mode=EXCLUDED.mode RETURNING id",
                trace_id, mode, int(seed), pack_hash, json.dumps(request_json, ensure_ascii=False),
            )
            return int(row[0]) if row else None
    except Exception:
        return None

async def _db_update_run_response(run_id: Optional[int], response_json: Dict[str, Any], metrics_json: Dict[str, Any]) -> None:
    if not run_id:
        return
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute("UPDATE run SET response_json=$1, metrics_json=$2 WHERE id=$3", json.dumps(response_json, ensure_ascii=False), json.dumps(metrics_json, ensure_ascii=False), int(run_id))
    except Exception:
        return

async def _db_insert_icw_log(run_id: Optional[int], pack_hash: Optional[str], budget_tokens: int, scores_json: Dict[str, Any]) -> None:
    if not (run_id and pack_hash):
        return
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO icw_log(run_id, pack_hash, budget_tokens, scores_json) VALUES($1,$2,$3,$4)",
                int(run_id), pack_hash, int(budget_tokens), json.dumps(scores_json, ensure_ascii=False),
            )
    except Exception:
        return

async def _db_insert_tool_call(run_id: Optional[int], name: str, seed: int, args_json: Dict[str, Any], result_json: Optional[Dict[str, Any]], duration_ms: Optional[int] = None) -> None:
    if not run_id:
        return
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO tool_call(run_id, name, seed, args_json, result_json, duration_ms) VALUES($1,$2,$3,$4,$5,$6)",
                int(run_id), name, int(seed), json.dumps(args_json, ensure_ascii=False), json.dumps(result_json or {}, ensure_ascii=False), duration_ms,
            )
    except Exception:
        return
def _normalize_dict_body(body: Any, expected_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Accept either a dict or a JSON string and return a dict shaped by expected_schema.
    Never raise on bad input; return {} as a safe default.
    """
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        try:
            return JSONParser().parse(body, expected_schema or {})
        except Exception:
            return {}
    return {}


QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11434")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen3:32b-instruct-q4_K_M")
GPTOSS_BASE_URL = os.getenv("GPTOSS_BASE_URL", "http://localhost:11435")
GPTOSS_MODEL_ID = os.getenv("GPTOSS_MODEL_ID", "gpt-oss:20b-q5_K_M")
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "8192"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
ENABLE_WEBSEARCH = os.getenv("ENABLE_WEBSEARCH", "false").lower() == "true"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
MCP_HTTP_BRIDGE_URL = os.getenv("MCP_HTTP_BRIDGE_URL")  # e.g., http://host.docker.internal:9999
EXECUTOR_BASE_URL = os.getenv("EXECUTOR_BASE_URL")  # http://executor:8081
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen")  # qwen | gptoss
ENABLE_DEBATE = os.getenv("ENABLE_DEBATE", "true").lower() == "true"
MAX_DEBATE_TURNS = int(os.getenv("MAX_DEBATE_TURNS", "1"))
AUTO_EXECUTE_TOOLS = os.getenv("AUTO_EXECUTE_TOOLS", "true").lower() == "true"
# Alias legacy flag to canonical; do not document legacy externally
ALLOW_TOOL_EXECUTION = AUTO_EXECUTE_TOOLS if os.getenv("ALLOW_TOOL_EXECUTION") is None else (os.getenv("ALLOW_TOOL_EXECUTION", "true").lower() == "true")
STREAM_CHUNK_SIZE_CHARS = int(os.getenv("STREAM_CHUNK_SIZE_CHARS", "0"))
STREAM_CHUNK_INTERVAL_MS = int(os.getenv("STREAM_CHUNK_INTERVAL_MS", "50"))
JOBS_RAG_INDEX = os.getenv("JOBS_RAG_INDEX", "true").lower() == "true"
# Defensive bounds
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "40000"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "50"))
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "6"))

# RAG configuration (pgvector)
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RAG_CACHE_TTL_SEC = int(os.getenv("RAG_CACHE_TTL_SEC", "300"))

# Optional external tool services (set URLs to enable)
COMFYUI_API_URL = os.getenv("COMFYUI_API_URL")  # e.g., http://comfyui:8188
COMFYUI_API_URLS = [u.strip() for u in os.getenv("COMFYUI_API_URLS", "").split(",") if u.strip()]
COMFYUI_REPLICAS = int(os.getenv("COMFYUI_REPLICAS", "1"))
SCENE_SUBMIT_CONCURRENCY = int(os.getenv("SCENE_SUBMIT_CONCURRENCY", "4"))
SCENE_MAX_BATCH_FRAMES = int(os.getenv("SCENE_MAX_BATCH_FRAMES", "2"))
RESUME_MAX_RETRIES = int(os.getenv("RESUME_MAX_RETRIES", "3"))
COMFYUI_API_URLS = [u.strip() for u in os.getenv("COMFYUI_API_URLS", "").split(",") if u.strip()]
SCENE_SUBMIT_CONCURRENCY = int(os.getenv("SCENE_SUBMIT_CONCURRENCY", "4"))
XTTS_API_URL = os.getenv("XTTS_API_URL")      # e.g., http://xtts:8020
WHISPER_API_URL = os.getenv("WHISPER_API_URL")# e.g., http://whisper:9090
FACEID_API_URL = os.getenv("FACEID_API_URL")  # e.g., http://faceid:7000
MUSIC_API_URL = os.getenv("MUSIC_API_URL")    # e.g., http://musicgen:7860
VLM_API_URL = os.getenv("VLM_API_URL")        # e.g., http://vlm:8050
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")  # optional external workflow orchestration
ENABLE_N8N = os.getenv("ENABLE_N8N", "false").lower() == "true"
ASSEMBLER_API_URL = os.getenv("ASSEMBLER_API_URL")  # http://assembler:9095
TEACHER_API_URL = os.getenv("TEACHER_API_URL", "http://teacher:8097")
WRAPPER_CONFIG_PATH = os.getenv("WRAPPER_CONFIG_PATH", "/workspace/configs/wrapper_config.json")
ICW_API_URL = os.getenv("ICW_API_URL", "http://icw:8085")
ICW_DISABLE = os.getenv("ICW_DISABLE", "0") == "1"
WRAPPER_CONFIG: Dict[str, Any] = {}
WRAPPER_CONFIG_HASH: Optional[str] = None
DRT_API_URL = os.getenv("DRT_API_URL", "http://drt:8086")
STATE_DIR = os.path.join(UPLOAD_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)
ARTIFACT_SHARD_BYTES = int(os.getenv("ARTIFACT_SHARD_BYTES", "200000"))
ARTIFACT_LATEST_ONLY = os.getenv("ARTIFACT_LATEST_ONLY", "true").lower() == "true"
SPOOL_RAM_LIMIT = int(os.getenv("SPOOL_RAM_LIMIT", "524288"))
SPOOL_SPILL_THRESHOLD = int(os.getenv("SPOOL_SPILL_THRESHOLD", "262144"))
EVENT_BUFFER_MAX = int(os.getenv("EVENT_BUFFER_MAX", "100"))
ARTIFACT_SHARD_BYTES = int(os.getenv("ARTIFACT_SHARD_BYTES", "200000"))
ARTIFACT_LATEST_ONLY = os.getenv("ARTIFACT_LATEST_ONLY", "true").lower() == "true"


def _sha256_bytes(b: bytes) -> str:
    import hashlib as _hl
    return _hl.sha256(b).hexdigest()


def _load_wrapper_config() -> None:
    global WRAPPER_CONFIG, WRAPPER_CONFIG_HASH
    try:
        if os.path.exists(WRAPPER_CONFIG_PATH):
            with open(WRAPPER_CONFIG_PATH, "rb") as f:
                data = f.read()
            WRAPPER_CONFIG_HASH = f"sha256:{_sha256_bytes(data)}"
            try:
                WRAPPER_CONFIG = json.loads(data.decode("utf-8"))
            except Exception:
                WRAPPER_CONFIG = {}
        else:
            WRAPPER_CONFIG = {}
            WRAPPER_CONFIG_HASH = None
    except Exception:
        WRAPPER_CONFIG = {}
        WRAPPER_CONFIG_HASH = None


_load_wrapper_config()


# removed: robust_json_loads — use JSONParser().parse with explicit expected structures everywhere


# No Pydantic. All request bodies are plain dicts validated by helpers.


app = FastAPI(title="Void Orchestrator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
@app.middleware("http")
async def global_cors_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return StreamingResponse(content=iter(()), status_code=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "false",
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Max-Age": "86400",
            "Access-Control-Allow-Private-Network": "true",
            "Connection": "close",
        })
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Credentials"] = "false"
    resp.headers["Access-Control-Expose-Headers"] = "*"
    resp.headers["Access-Control-Max-Age"] = "86400"
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


# In-memory job cache (DB is source of truth)
_jobs_store: Dict[str, Dict[str, Any]] = {}
_job_endpoint: Dict[str, str] = {}
_comfy_load: Dict[str, int] = {}
COMFYUI_BACKOFF_MS = int(os.getenv("COMFYUI_BACKOFF_MS", "250"))
COMFYUI_BACKOFF_MAX_MS = int(os.getenv("COMFYUI_BACKOFF_MAX_MS", "4000"))
COMFYUI_MAX_RETRIES = int(os.getenv("COMFYUI_MAX_RETRIES", "6"))
try:
    _comfy_sem = asyncio.Semaphore(max(1, SCENE_SUBMIT_CONCURRENCY))
except Exception:
    _comfy_sem = None
_films_mem: Dict[str, Dict[str, Any]] = {}
_characters_mem: Dict[str, Dict[str, Any]] = {}
_scenes_mem: Dict[str, Dict[str, Any]] = {}

def _uri_from_upload_path(path: str) -> str:
    rel = os.path.relpath(path, UPLOAD_DIR).replace("\\", "/")
    return f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel}" if PUBLIC_BASE_URL else f"/uploads/{rel}"

def _write_text_atomic(path: str, text: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return {"uri": _uri_from_upload_path(path), "hash": f"sha256:{_sha256_bytes(text.encode('utf-8'))}"}

def _write_json_atomic(path: str, obj: Any) -> Dict[str, Any]:
    import json as _j
    return _write_text_atomic(path, _j.dumps(obj, ensure_ascii=False, separators=(",", ":")))


@app.get("/capabilities.json")
async def capabilities():
    return {
        "openai_compat": True,
        "endpoints": {
            "run": "/run",
            "film": {
                "plan": "/film/plan",
                "breakdown": "/film/breakdown",
                "storyboard": "/film/storyboard",
                "animatic": "/film/animatic",
                "final": "/film/final",
                "qc": "/film/qc",
                "export": "/film/export"
            },
            "ablation": "/ablate",
            "teacher": {
                "enable": "/teacher/trace.enable",
                "flush": "/teacher/trace.flush",
                "ingest": "/teacher/trainset.ingest"
            }
        },
        "tools": ["film.run","web_search","source_fetch","metasearch.fuse"],
        "versions": {
            "icw": "1.0.0",
            "film": "2.0.0",
            "ablation": "1.0.0",
            "teacher": "1.0.0"
        },
        "config_hash": WRAPPER_CONFIG_HASH
    }


def build_ollama_payload(messages: List[Dict[str, Any]], model: str, num_ctx: int, temperature: float) -> Dict[str, Any]:
    rendered: List[str] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            tool_name = m.get("name") or "tool"
            rendered.append(f"tool[{tool_name}]: {m.get('content')}")
        else:
            content_val = m.get("content")
            if isinstance(content_val, list):
                text_parts: List[str] = []
                for part in content_val:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                content = "\n".join(text_parts)
            else:
                content = content_val if content_val is not None else ""
            rendered.append(f"{role}: {content}")
    prompt = "\n".join(rendered)
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
        },
    }


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # crude approximation: 1 token ~ 4 chars
    return max(1, (len(text) + 3) // 4)


def estimate_usage(messages: List[Dict[str, Any]], completion_text: str) -> Dict[str, int]:
    prompt_text = "\n".join([(m.get("content") or "") for m in messages])
    prompt_tokens = estimate_tokens_from_text(prompt_text)
    completion_tokens = estimate_tokens_from_text(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _icw_pack(messages: List[Dict[str, Any]], seed: int, budget_tokens: int = 3500) -> Dict[str, Any]:
    # Inline, deterministic packer with simple multi-signal scoring and graded budget allocation.
    # No network; JSON-only; scores rounded to 1e-6; stable tie-break by sha256.
    try:
        import hashlib as _hl
        def _round6(x: float) -> float:
            return float(f"{float(x):.6f}")
        def _tok(s: str) -> List[str]:
            import re as _re
            return [w for w in _re.findall(r"[a-z0-9]{3,}", (s or "").lower())]
        def _jacc(a: List[str], b: List[str]) -> float:
            if not a or not b:
                return 0.0
            sa, sb = set(a), set(b)
            if not sa or not sb:
                return 0.0
            inter = len(sa & sb)
            uni = len(sa | sb)
            return inter / float(uni or 1)
        # Query = latest user message
        query_text = ""
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip():
                query_text = m.get("content").strip()
                break
        qtok = _tok(query_text)
        # Candidates = each message content (with role), scored
        items: List[Dict[str, Any]] = []
        owner_counts: Dict[str, int] = {}
        for idx, m in enumerate(messages):
            role = m.get("role") or "user"
            content = str(m.get("content") or "")
            if not content:
                continue
            toks = _tok(content)
            topical = _jacc(toks, qtok)
            # Simple role-based authority prior
            authority = {"system": 0.9, "tool": 0.7, "assistant": 0.5, "user": 0.4}.get(role, 0.4)
            # Recency prior: later messages higher
            recency = (idx + 1) / float(len(messages))
            # Diversity proxy by role owner
            owner = role
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
            items.append({"role": role, "content": content, "topical": topical, "authority": authority, "recency": recency, "owner": owner})
        # Diversity score needs owner counts
        for it in items:
            own = it.get("owner")
            freq = owner_counts.get(own, 1)
            it["diversity"] = 1.0 / float(freq)
        # Composite score (weights): topical 0.40, authority 0.25, recency 0.20, diversity 0.15
        for it in items:
            s = 0.40 * it["topical"] + 0.25 * it["authority"] + 0.20 * it["recency"] + 0.15 * it["diversity"]
            it["score"] = _round6(s)
            it["sha"] = _hl.sha256((it.get("role") + "\n" + it.get("content")).encode("utf-8")).hexdigest()
        # Stable sort: score desc, topical desc, authority desc, recency desc, sha asc
        items.sort(key=lambda x: (-x["score"], -x["topical"], -x["authority"], -x["recency"], x["sha"]))
        # Budget allocator: try full texts in order, then degrade to summaries (first N chars) if overflow
        def _fit(texts: List[str], budget: int) -> str:
            acc: List[str] = []
            used = 0
            for t in texts:
                need = estimate_tokens_from_text(t)
                if used + need <= budget:
                    acc.append(t)
                    used += need
                else:
                    # try a shorter summary tier
                    short = t[: max(200, int(len(t) * 0.25))]
                    need2 = estimate_tokens_from_text(short)
                    if used + need2 <= budget:
                        acc.append(short)
                        used += need2
                if used >= budget:
                    break
            return "\n\n".join(acc)
        ranked_texts = [f"{it['role']}: {it['content']}" for it in items]
        pack = _fit(ranked_texts, budget_tokens)
        ph = _hl.sha256(pack.encode("utf-8")).hexdigest()
        scores_summary = {
            "selected": len([1 for _ in ranked_texts]),
            "dup_rate": _round6(0.0),
            "independence_index": _round6(min(1.0, len(owner_counts.keys()) / 6.0)),
        }
        return {"pack": pack, "hash": f"sha256:{ph}", "budget_tokens": budget_tokens, "estimated_tokens": estimate_tokens_from_text(pack), "scores_summary": scores_summary}
    except Exception:
        return {"pack": "", "hash": None, "budget_tokens": budget_tokens, "estimated_tokens": 0, "scores_summary": {}}


def merge_usages(usages: List[Optional[Dict[str, int]]]) -> Dict[str, int]:
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usages:
        if not u:
            continue
        out["prompt_tokens"] += int(u.get("prompt_tokens", 0))
        out["completion_tokens"] += int(u.get("completion_tokens", 0))
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def _save_base64_file(b64: str, suffix: str) -> str:
    import base64, uuid
    raw = base64.b64decode(b64)
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(raw)
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{filename}"
    return f"/uploads/{filename}"


def extract_attachments_from_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    attachments: List[Dict[str, Any]] = []
    normalized: List[Dict[str, Any]] = []
    for m in messages:
        content_val = m.get("content")
        if isinstance(content_val, list):
            text_parts: List[str] = []
            for part in content_val:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(str(part.get("text", "")))
                elif ptype in ("input_image", "image"):
                    iu = part.get("image_url") or {}
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    b64 = part.get("image_base64")
                    final_url = None
                    if b64:
                        final_url = _save_base64_file(b64, ".png")
                    elif url:
                        final_url = url
                    if final_url:
                        attachments.append({"type": "image", "url": final_url})
                elif ptype in ("input_audio", "audio"):
                    au = part.get("audio_url") or {}
                    url = au.get("url") if isinstance(au, dict) else au
                    b64 = part.get("audio_base64")
                    final_url = None
                    if b64:
                        final_url = _save_base64_file(b64, ".wav")
                    elif url:
                        final_url = url
                    if final_url:
                        attachments.append({"type": "audio", "url": final_url})
                elif ptype in ("input_video", "video"):
                    vu = part.get("video_url") or {}
                    url = vu.get("url") if isinstance(vu, dict) else vu
                    b64 = part.get("video_base64")
                    final_url = None
                    if b64:
                        final_url = _save_base64_file(b64, ".mp4")
                    elif url:
                        final_url = url
                    if final_url:
                        attachments.append({"type": "video", "url": final_url})
                elif ptype in ("input_file", "file"):
                    fu = part.get("file_url") or {}
                    url = fu.get("url") if isinstance(fu, dict) else fu
                    b64 = part.get("file_base64")
                    name = part.get("name") or "file.bin"
                    suffix = "." + name.split(".")[-1] if "." in name else ".bin"
                    final_url = None
                    if b64:
                        final_url = _save_base64_file(b64, suffix)
                    elif url:
                        final_url = url
                    if final_url:
                        attachments.append({"type": "file", "url": final_url, "name": name})
            merged_text = "\n".join(tp for tp in text_parts if tp)
            normalized.append({"role": m.get("role"), "content": merged_text or None, "name": m.get("name"), "tool_call_id": m.get("tool_call_id"), "tool_calls": m.get("tool_calls")})
        else:
            normalized.append(m)
    return normalized, attachments


# ---------- RAG (pgvector) ----------
pg_pool: Optional[asyncpg.pool.Pool] = None
_embedder: Optional[SentenceTransformer] = None
_rag_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


async def get_pg_pool() -> Optional[asyncpg.pool.Pool]:
    global pg_pool
    if pg_pool is not None:
        return pg_pool
    if not (POSTGRES_HOST and POSTGRES_DB and POSTGRES_USER and POSTGRES_PASSWORD):
        return None
    pg_pool = await asyncpg.create_pool(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        min_size=1,
        max_size=10,
    )
    async with pg_pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_docs (
              id BIGSERIAL PRIMARY KEY,
              path TEXT,
              chunk TEXT,
              embedding vector(384)
            );
            """
        )
        # Ensure pgvector extension and indexes exist (safe to run if extension already installed)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            pass
        try:
            # pgvector opclass names differ by version: use *_ops form
            await conn.execute("CREATE INDEX IF NOT EXISTS rag_docs_embedding_idx ON rag_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
        except Exception:
            # Fallback: HNSW with L2 opclass
            try:
                await conn.execute("CREATE INDEX IF NOT EXISTS rag_docs_embedding_hnsw_idx ON rag_docs USING hnsw (embedding vector_l2_ops);")
            except Exception:
                pass
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              prompt_id TEXT,
              status TEXT,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW(),
              workflow JSONB,
              result JSONB,
              error TEXT
            );
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS jobs_status_updated_idx ON jobs (status, updated_at DESC);")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_checkpoints (
              id BIGSERIAL PRIMARY KEY,
              job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              data JSONB
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS films (
              id TEXT PRIMARY KEY,
              title TEXT,
              synopsis TEXT,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW(),
              metadata JSONB
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
              id TEXT PRIMARY KEY,
              film_id TEXT REFERENCES films(id) ON DELETE CASCADE,
              name TEXT,
              description TEXT,
              reference_data JSONB,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scenes (
              id TEXT PRIMARY KEY,
              film_id TEXT REFERENCES films(id) ON DELETE CASCADE,
              index_num INT,
              prompt TEXT,
              plan JSONB,
              status TEXT,
              job_id TEXT,
              assets JSONB,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
    return pg_pool


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


async def rag_index_dir(root: str = "/workspace", glob_exts: Optional[List[str]] = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    import glob as _glob
    import os as _os
    pool = await get_pg_pool()
    if pool is None:
        return {"error": "pgvector not configured"}
    exts = glob_exts or ["*.md", "*.py", "*.ts", "*.tsx", "*.js", "*.json", "*.txt"]
    files: List[str] = []
    for ext in exts:
        files.extend(_glob.glob(_os.path.join(root, "**", ext), recursive=True))
    embedder = get_embedder()
    total_chunks = 0
    async with pool.acquire() as conn:
        for fp in files[:5000]:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            i = 0
            length = len(content)
            while i < length:
                chunk = content[i : i + chunk_size]
                i += max(1, chunk_size - chunk_overlap)
                vec = embedder.encode([chunk])[0]
                await conn.execute("INSERT INTO rag_docs (path, chunk, embedding) VALUES ($1, $2, $3)", fp.replace(root + "/", ""), chunk, list(vec))
                total_chunks += 1
    return {"indexed_files": len(files), "chunks": total_chunks}


async def rag_search(query: str, k: int = 8) -> List[Dict[str, Any]]:
    now = time.time()
    key = f"{query}::{k}"
    cached = _rag_cache.get(key)
    if cached and (now - cached[0] <= RAG_CACHE_TTL_SEC):
        return cached[1]
    pool = await get_pg_pool()
    if pool is None:
        return []
    vec = get_embedder().encode([query])[0]
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT path, chunk FROM rag_docs ORDER BY embedding <=> $1 LIMIT $2", list(vec), k)
        results = [{"path": r["path"], "chunk": r["chunk"]} for r in rows]
        # Apply hygiene: de-dup and newest-first (ts may be absent)
        try:
            from .rag.hygiene import rag_filter as _rf
            # map to {title,text,ts,url}
            mapped = [{"title": it.get("path"), "text": it.get("chunk"), "ts": 0, "url": it.get("path")} for it in results]
            clean = _rf(mapped)
            # map back to original structure but keep order
            results = [{"path": c.get("title"), "chunk": c.get("text") } for c in clean]
        except Exception:
            pass
        _rag_cache[key] = (now, results)
        return results


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=2))
async def call_ollama(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            # Keep models warm across requests
            ppayload = dict(payload)
            opts = dict(ppayload.get("options") or {})
            opts["keep_alive"] = opts.get("keep_alive") or "24h"
            ppayload["options"] = opts
            resp = await client.post(f"{base_url}/api/generate", json=ppayload)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and ("prompt_eval_count" in data or "eval_count" in data):
                usage = {
                    "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
                    "completion_tokens": int(data.get("eval_count", 0) or 0),
                }
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                data["_usage"] = usage
            return data
        except httpx.HTTPError as e:
            return {"error": str(e), "_base_url": base_url}


async def serpapi_google_search(queries: List[str], max_results: int = 5) -> str:
    if not SERPAPI_API_KEY:
        return ""
    aggregate = []
    async with httpx.AsyncClient(timeout=30) as client:
        for q in queries:
            try:
                params = {
                    "engine": "google",
                    "q": q,
                    "num": max_results,
                    "api_key": SERPAPI_API_KEY,
                }
                r = await client.get("https://serpapi.com/search.json", params=params)
                r.raise_for_status()
                data = r.json()
                organic = data.get("organic_results", [])
                for item in organic[:max_results]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    aggregate.append(f"- {title}\n{snippet}\n{link}")
            except Exception:
                continue
    return "\n".join(aggregate[: max_results * max(1, len(queries))])


def _rrf_fuse(results_by_engine: Dict[str, List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    # Reciprocal Rank Fusion + CombMNZ with stable tie-break
    scores: Dict[str, float] = {}
    freq: Dict[str, int] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for eng, items in results_by_engine.items():
        for rank, it in enumerate(items, start=1):
            key = it.get("link") or it.get("url") or it.get("id") or f"{eng}:{rank}"
            rr = 1.0 / float(k + rank)
            scores[key] = scores.get(key, 0.0) + rr
            freq[key] = freq.get(key, 0) + 1
            if key not in meta:
                meta[key] = {"title": it.get("title"), "link": it.get("link") or it.get("url"), "snippet": it.get("snippet")}
    # Combine: CombMNZ = (sum of scores) * freq; round to 1e-6 and tie-break deterministically
    def _auth(x: Dict[str, Any]) -> float:
        return 0.0
    def _rec(x: Dict[str, Any]) -> float:
        return 0.0
    def _sha(x: str) -> str:
        import hashlib as _h
        return _h.sha256(x.encode("utf-8")).hexdigest()
    items: List[Dict[str, Any]] = []
    for key, sc in scores.items():
        m = meta.get(key) or {}
        comb = float(f"{(sc * float(freq.get(key, 1))):.6f}")
        items.append({"key": key, "score": comb, "rrf": float(f"{sc:.6f}"), "freq": int(freq.get(key, 1)), **m})
    items.sort(key=lambda r: (-r["score"], -_auth(r), -_rec(r), _sha(r.get("key") or "")))
    return items


async def call_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not MCP_HTTP_BRIDGE_URL:
        return {"error": "MCP bridge URL not configured"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                MCP_HTTP_BRIDGE_URL.rstrip("/") + "/call",
                json={"name": name, "arguments": arguments or {}},
            )
            r.raise_for_status()
            return r.json()
    except Exception as ex:
        return {"error": str(ex)}


def to_openai_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for idx, c in enumerate(tool_calls):
        name = c.get("name") or "tool"
        args = c.get("arguments") or {}
        converted.append({
            "id": f"call_{idx+1}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return converted


async def propose_search_queries(messages: List[Dict[str, Any]]) -> List[str]:
    guidance = (
        "Given the conversation, propose up to 3 concise web search queries that would most improve the answer. "
        "Return each query on its own line with no extra text."
    )
    prompt_messages = messages + [{"role": "user", "content": guidance}]
    payload = build_ollama_payload(prompt_messages, QWEN_MODEL_ID, DEFAULT_NUM_CTX, DEFAULT_TEMPERATURE)
    try:
        result = await call_ollama(QWEN_BASE_URL, payload)
        text = result.get("response", "")
        lines = [ln.strip("- ") for ln in text.splitlines() if ln.strip()]
        # take up to 3 non-empty queries
        return lines[:3]
    except Exception:
        return []


def meta_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_preface = (
        "You are part of a two-model team with explicit roles: a Planner and Executors. "
        "Planner decomposes the task, chooses tools, and requests relevant evidence. Executors produce solutions and critiques. "
        "You execute tools yourself. Never ask the user to run code or call tools; do not output scripts for the user to execute. "
        "Be precise, show working when non-trivial, and when a tool can fulfill the request, invoke it directly. "
        "Absolute rule: Never use SQLAlchemy; use asyncpg with pooling and raw SQL for PostgreSQL. "
        "Critically: choose tools SEMANTICALLY by mapping the user's requested outcome to tool capabilities (no keyword triggers). "
        "If any available tool can directly produce the requested artifact, invoke that tool; otherwise provide a concrete solution or plan. "
        "Never refuse actionable requests. Do not apologize, do not say you can't or that it's not supported. If something seems constrained, state constraints briefly and proceed with best-effort steps using reasonable defaults. "
        "Ask at most one concise clarifying message if a blocking parameter is missing; include up to three targeted questions in that single message, then proceed. "
        "When the user provides reference media (audio, images, video, files), treat them as style/constraint references. "
        "Do NOT reuse source media as-is unless explicitly instructed; transform or condition on them to match style, timing/beat, or visual attributes (e.g., hair/face), respecting the user's constraints. "
        "Always surface IDs and errors produced by tools: include film_id, job_id(s), prompt_id(s) when available, and concise error/traceback summaries. "
        "Return the complete Markdown answer; do not omit the main body when tools are used. "
        "If you clearly see a small, relevant fix or improvement (e.g., a failing tool parameter or obvious config issue), you may add a short 'Suggestions' section with up to 2 concise bullet points at the end. Do not scope creep."
    )
    out = [{"role": "system", "content": system_preface + "\nHARD BAN: Never import or recommend pydantic, sqlalchemy, or any ORM/CSV/Parquet libs; use asyncpg + JSON/NDJSON only."}]
    # Provide additional guidance for high-fidelity video preferences without relying on keywords at routing time
    out.append({"role": "system", "content": (
        "When generating videos, reasonable defaults are: duration<=10s, resolution=1920x1080, 24fps, language=en, neutral voice, audio ON, subtitles OFF. "
        "If user implies higher fidelity (e.g., 4K/60fps), set resolution=3840x2160 and fps=60, enable interpolation and upscale accordingly."
    )})
    return out + messages


def merge_responses(request: Dict[str, Any], qwen_text: str, gptoss_text: str) -> str:
    return (
        "Committee Synthesis:\n\n"
        "Qwen perspective:\n" + qwen_text.strip() + "\n\n"
        "GPT-OSS perspective:\n" + gptoss_text.strip() + "\n\n"
        "Final answer (synthesize both; prefer correctness and specificity):\n"
    )


def build_tools_section(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return ""
    try:
        return "Available tools (JSON schema):\n" + json.dumps(tools, indent=2)
    except Exception:
        return ""


def get_builtin_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "film.run",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "duration_s": {"type": "integer"},
                        "seed": {"type": "integer"},
                        "style_refs": {"type": "array", "items": {"type": "string"}},
                        "character_images": {"type": "array", "items": {"type": "object"}},
                        "res": {"type": "string"},
                        "refresh": {"type": "integer"},
                        "base_fps": {"type": "integer"},
                        "codec": {"type": "string"},
                        "container": {"type": "string"},
                        "post": {"type": "object"},
                        "audio": {"type": "object"},
                        "safety": {"type": "object"}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "source_fetch",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "metasearch.fuse",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "k": {"type": "integer"}}, "required": ["q"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "image.dispatch",
                "parameters": {"type": "object", "properties": {"mode": {"type": "string"}, "prompt": {"type": "string"}, "scale": {"type": "integer"}}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "music.dispatch",
                "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tts_speak",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "voice": {"type": "string"}}, "required": ["text"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "research.run",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "scope": {"type": "string"}}, "required": ["query"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "code.super_loop",
                "parameters": {"type": "object", "properties": {"task": {"type": "string"}, "repo_root": {"type": "string"}}, "required": ["task"]}
            }
        }
    ]


def _detect_video_intent(text: str) -> bool:
    if not text:
        return False
    s = text.lower()
    keywords = (
        "video", "film", "movie", "scene", "4k", "8k", "60fps", "fps", "frame rate",
        "minute", "minutes", "sec", "second", "seconds"
    )
    return any(k in s for k in keywords)


def _derive_movie_prefs_from_text(text: str) -> Dict[str, Any]:
    prefs: Dict[str, Any] = {}
    s = (text or "").lower()
    # resolution
    if "4k" in s:
        prefs["resolution"] = "3840x2160"
    elif "8k" in s:
        prefs["resolution"] = "7680x4320"
    # fps
    import re as _re
    m_fps = _re.search(r"(\d{2,3})\s*fps", s)
    if m_fps:
        try:
            prefs["fps"] = float(m_fps.group(1))
        except Exception:
            pass
    elif "60fps" in s or "60 fps" in s:
        prefs["fps"] = 60.0
    # explicit duration in minutes/seconds
    m_min = _re.search(r"(\d+)\s*minute", s)
    m_sec = _re.search(r"(\d+)\s*sec", s)
    if m_min:
        try:
            prefs["duration_seconds"] = float(int(m_min.group(1)) * 60)
        except Exception:
            pass
    elif m_sec:
        try:
            prefs["duration_seconds"] = float(int(m_sec.group(1)))
        except Exception:
            pass
    # infer minimal coherent duration if not specified
    if "duration_seconds" not in prefs:
        words = len([w for w in s.replace("\n", " ").split(" ") if w.strip()])
        # category heuristics (minimal coherent durations)
        if any(k in s for k in ("meme", "clip", "gif")):
            inferred = 10.0
        elif any(k in s for k in ("trailer", "teaser")):
            inferred = 60.0
        elif any(k in s for k in ("ad", "advert", "commercial")):
            inferred = 30.0
        elif "music video" in s:
            inferred = 210.0  # ~3.5 minutes
        elif "short film" in s:
            inferred = 300.0  # ~5 minutes
        elif any(k in s for k in ("episode", "tv episode")):
            inferred = 1500.0  # ~25 minutes
        elif any(k in s for k in ("feature film",)):
            inferred = 5400.0  # ~90 minutes (minimal feature)
        elif "documentary" in s:
            inferred = 900.0  # ~15 minutes minimal doc short
        else:
            # scale with description complexity
            if words >= 1500:
                inferred = 3600.0
            elif words >= 600:
                inferred = 600.0
            elif words >= 150:
                inferred = 60.0
            else:
                inferred = 20.0
        prefs["duration_seconds"] = inferred
    # sensible defaults
    prefs.setdefault("resolution", "1920x1080")
    prefs.setdefault("fps", 24.0)
    return prefs

def merge_tool_schemas(client_tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    builtins = get_builtin_tools_schema()
    if not client_tools:
        return builtins
    # merge by function name
    seen = {((t.get("function") or {}).get("name") or t.get("name")) for t in client_tools}
    out = list(client_tools)
    for t in builtins:
        name = (t.get("function") or {}).get("name") or t.get("name")
        if name not in seen:
            out.append(t)
    return out

async def planner_produce_plan(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], temperature: float) -> Tuple[str, List[Dict[str, Any]]]:
    # Treat any 'qwen*' value as Qwen route (e.g., 'qwen3')
    use_qwen = str(PLANNER_MODEL or "qwen").lower().startswith("qwen")
    planner_id = QWEN_MODEL_ID if use_qwen else GPTOSS_MODEL_ID
    planner_base = QWEN_BASE_URL if use_qwen else GPTOSS_BASE_URL
    guide = (
        "You are the Planner. Produce a short step-by-step plan and propose up to 3 tool invocations YOU will execute.\n"
        "Never instruct the user to run code or call tools. Do not emit shell/python snippets for the user; invoke tools yourself.\n"
        "Decide SEMANTICALLY when to invoke tools: map the user's requested outcome to tool capabilities.\n"
        "If any tool can directly produce the user's requested artifact (e.g., video/film creation), prefer that tool (e.g., make_movie).\n"
        "Ask 1-3 clarifying questions ONLY if blocking details are missing (e.g., duration, style, language, target resolution).\n"
        "If not blocked, proceed and choose reasonable defaults: duration<=10s for short clips, 1920x1080, 24fps, language=en, neutral voice.\n"
        "Return strict JSON with keys: plan (string), tool_calls (array of {name: string, arguments: object}).\n"
        "Absolute rule: Never propose or use SQLAlchemy in code or tools. Use asyncpg with pooled connections and raw SQL for PostgreSQL."
    )
    all_tools = merge_tool_schemas(tools)
    tool_info = build_tools_section(all_tools)
    plan_messages = messages + [{"role": "user", "content": guide + ("\n" + tool_info if tool_info else "")}]
    payload = build_ollama_payload(plan_messages, planner_id, DEFAULT_NUM_CTX, temperature)
    result = await call_ollama(planner_base, payload)
    text = result.get("response", "").strip()
    # Use custom parser to normalise the planner JSON
    parser = JSONParser()
    expected = {"plan": str, "tool_calls": [ {"name": str, "arguments": dict} ]}
    parsed = parser.parse(text, expected)
    plan = parsed.get("plan", "")
    tool_calls = parsed.get("tool_calls", []) or []
    # If planner proposes nothing but user intent implies video/film, synthesize make_movie call
    if not tool_calls:
        # find latest user message
        last_user = ""
        for m in reversed(messages):
            if (isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()):
                last_user = m.get("content").strip()
                break
        if _detect_video_intent(last_user):
            prefs = _derive_movie_prefs_from_text(last_user)
            # Prefer unified Film‑2 tool
            tool_calls = [{"name": "film.run", "arguments": {"title": "Untitled", "duration_s": int(prefs.get("duration_seconds", 10)), "res": prefs.get("resolution", "1920x1080"), "refresh": int(prefs.get("fps", 24)), "base_fps": 30}}]
    else:
        # If planner proposed film_create for a film request (but not make_movie), upgrade to make_movie
        last_user = ""
        for m in reversed(messages):
            if (isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content").strip()):
                last_user = m.get("content").strip()
                break
        if _detect_video_intent(last_user):
            has_run = any((tc.get("name") == "film.run") for tc in tool_calls if isinstance(tc, dict))
            if not has_run:
                # Replace any legacy film_* tools with a single film.run
                prefs = _derive_movie_prefs_from_text(last_user)
                tool_calls = [{"name": "film.run", "arguments": {"title": "Untitled", "duration_s": int(prefs.get("duration_seconds", 10)), "res": prefs.get("resolution", "1920x1080"), "refresh": int(prefs.get("fps", 24)), "base_fps": 30}}]
    return plan, tool_calls


async def execute_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name")
    raw_args = call.get("arguments") or {}
    # Hard-disable legacy Film-1 tools to enforce Film-2-only orchestration
    if name in ("film_create", "film_add_character", "film_add_scene", "film_status", "film_compile", "make_movie"):
        return {"name": name, "skipped": True, "reason": "film1_disabled"}
    # DB pool (may be None if PG not configured)
    pool = await get_pg_pool()
    # Normalize tool arguments using the custom parser and strong coercion
    # NOTE: Do not use json.loads here. Always use JSONParser with a schema so we never crash on
    # malformed/partial JSON or return unexpected shapes from LLM/tool output.
    def _normalize_tool_args(a: Any) -> Dict[str, Any]:
        parser = JSONParser()
        if isinstance(a, dict):
            out = dict(a)
        elif isinstance(a, str):
            expected = {
                "film_id": str,
                "title": str,
                "synopsis": str,
                "prompt": str,
                "characters": [ {"name": str, "description": str} ],
                "scenes": [ {"index_num": int, "prompt": str} ],
                "index_num": int,
                "duration_seconds": float,
                "duration": float,
                "resolution": str,
                "fps": float,
                "frame_rate": float,
                "style": str,
                "language": str,
                "voice": str,
                "audio_enabled": str,
                "audio_on": str,
                "subtitles_enabled": str,
                "subtitles_on": str,
                "metadata": dict,
                "references": dict,
                "reference_data": dict,
            }
            parsed = parser.parse(a, expected)
            # If nothing meaningful parsed, treat as synopsis/prompt
            if not any(str(parsed.get(k) or "").strip() for k in ("title","synopsis","prompt")):
                out = {"synopsis": a}
            else:
                out = parsed
        else:
            out = {}
        # Synonyms → canonical
        if out.get("duration_seconds") is None and out.get("duration") is not None:
            out["duration_seconds"] = out.get("duration")
        if out.get("fps") is None and out.get("frame_rate") is not None:
            out["fps"] = out.get("frame_rate")
        if out.get("reference_data") is None and out.get("references") is not None:
            out["reference_data"] = out.get("references")
        # Coerce types
        def _to_float(v):
            try:
                return float(v)
            except Exception:
                return None
        def _to_int(v):
            try:
                return int(v)
            except Exception:
                return None
        if out.get("fps") is not None:
            f = _to_float(out.get("fps"))
            if f is not None:
                out["fps"] = f
        if out.get("duration_seconds") is not None:
            d = _to_float(out.get("duration_seconds"))
            if d is not None:
                out["duration_seconds"] = d
        if out.get("index_num") is not None:
            i = _to_int(out.get("index_num"))
            if i is not None:
                out["index_num"] = i
        # Booleans that might arrive as strings
        for key, syn in (("audio_enabled","audio_on"),("subtitles_enabled","subtitles_on")):
            val = out.get(key)
            if val is None:
                val = out.get(syn)
            if isinstance(val, str):
                low = val.strip().lower()
                if low in ("true","1","yes","on"): val = True
                elif low in ("false","0","no","off"): val = False
                else: val = None
            if isinstance(val, (bool,)):
                out[key] = bool(val)
        # Additional aliases often provided by clients
        if out.get("audio_enabled") is None and ("audio" in out):
            v = out.get("audio")
            if isinstance(v, str):
                vv = v.strip().lower(); v = True if vv in ("true","1","yes","on") else False if vv in ("false","0","no","off") else None
            if isinstance(v, (bool,)):
                out["audio_enabled"] = bool(v)
        if out.get("subtitles_enabled") is None and ("subtitles" in out):
            v = out.get("subtitles")
            if isinstance(v, str):
                vv = v.strip().lower(); v = True if vv in ("true","1","yes","on") else False if vv in ("false","0","no","off") else None
            if isinstance(v, (bool,)):
                out["subtitles_enabled"] = bool(v)
        # Normalize resolution synonyms
        res = out.get("resolution")
        if isinstance(res, str):
            r = res.lower()
            if r in ("4k","3840x2160","3840×2160"): out["resolution"] = "3840x2160"
            elif r in ("8k","7680x4320","7680×4320"): out["resolution"] = "7680x4320"
        return out

    args = _normalize_tool_args(raw_args)
    # Deterministic grouped dispatchers
    if name == "image.dispatch" and ALLOW_TOOL_EXECUTION:
        mode = (args.get("mode") or "gen").lower()
        if mode == "upscale":
            return {"name": "image.dispatch", "result": {"accepted": True, "op": "upscale", "params": {"scale": int(args.get("scale") or 2)}}}
        if mode == "edit" and COMFYUI_API_URL and ALLOW_TOOL_EXECUTION:
            async with httpx.AsyncClient(timeout=600) as client:
                r = await client.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json=args.get("workflow") or args)
                try:
                    r.raise_for_status()
                    return {"name": name, "result": r.json()}
                except Exception:
                    return {"name": name, "error": r.text}
        # default gen
        if COMFYUI_API_URL and ALLOW_TOOL_EXECUTION:
            async with httpx.AsyncClient(timeout=600) as client:
                r = await client.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json=args.get("workflow") or args)
                try:
                    r.raise_for_status()
                    return {"name": name, "result": r.json()}
                except Exception:
                    return {"name": name, "error": r.text}
        return {"name": name, "skipped": True, "reason": "image backend not configured"}
    if name == "music.dispatch" and MUSIC_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=1200) as client:
            r = await client.post(MUSIC_API_URL.rstrip("/") + "/generate", json=args)
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "research.run" and ALLOW_TOOL_EXECUTION:
        # Prefer local orchestrator; on failure, try DRT service if configured
        try:
            job_args = args if isinstance(args, dict) else {}
            try:
                import uuid as _uuid
                job_args.setdefault("job_id", _uuid.uuid4().hex)
            except Exception:
                pass
            result = run_research(job_args)
            if isinstance(result, dict):
                result.setdefault("job_id", job_args.get("job_id"))
            return {"name": name, "result": result}
        except Exception as ex:
            base = (DRT_API_URL or "").rstrip("/")
            if not base:
                return {"name": name, "error": str(ex)}
            try:
                async with httpx.AsyncClient(timeout=900) as client:
                    r = await client.post(base + "/research/run", json=args)
                    r.raise_for_status()
                    return {"name": name, "result": r.json()}
            except Exception as ex2:
                return {"name": name, "error": str(ex2)}
    if name == "code.super_loop" and ALLOW_TOOL_EXECUTION:
        # Build a local provider wrapper on top of Qwen
        class _LocalProvider:
            def __init__(self, base_url: str, model_id: str, num_ctx: int, temperature: float):
                self.base_url = base_url
                self.model_name = model_id
                self.num_ctx = num_ctx
                self.temperature = temperature
            def chat(self, prompt: str, max_tokens: int):
                payload = build_ollama_payload([{"role": "user", "content": prompt}], self.model_name, self.num_ctx, self.temperature)
                opts = dict(payload.get("options") or {})
                opts["num_predict"] = max(1, int(max_tokens or 900))
                payload["options"] = opts
                # reuse async call_ollama via blocking run
                import requests as _rq
                try:
                    r = _rq.post(self.base_url.rstrip("/") + "/api/generate", json=payload, timeout=60)
                    r.raise_for_status()
                    js = r.json()
                    return SimpleNamespace(text=str(js.get("response", "")), model_name=self.model_name)
                except Exception:
                    return SimpleNamespace(text="{}", model_name=self.model_name)
        repo_root = os.getenv("REPO_ROOT", "/workspace")
        step_tokens = int(os.getenv("CODE_LOOP_STEP_TOKENS", "900") or 900)
        prov = _LocalProvider(QWEN_BASE_URL, QWEN_MODEL_ID, DEFAULT_NUM_CTX, DEFAULT_TEMPERATURE)
        task = args.get("task") or ""
        env = run_super_loop(task=task, repo_root=repo_root, model=prov, step_tokens=step_tokens)
        return {"name": name, "result": env}
    if name == "web_search" and ENABLE_WEBSEARCH and SERPAPI_API_KEY and ALLOW_TOOL_EXECUTION:
        query = args.get("q") or args.get("query") or ""
        if not query:
            return {"name": name, "error": "missing query"}
        snippets = await serpapi_google_search([query], max_results=int(args.get("k", 5)))
        return {"name": name, "result": snippets}
    if name == "metasearch.fuse" and ALLOW_TOOL_EXECUTION:
        q = args.get("q") or ""
        if not q:
            return {"name": name, "error": "missing q"}
        k = int(args.get("k", 10))
        # Deterministic multi-engine placeholder. If SERPAPI available, use it to seed one engine; others are deterministic transforms
        engines: Dict[str, List[Dict[str, Any]]] = {}
        # google via serpapi (optional)
        if SERPAPI_API_KEY and ENABLE_WEBSEARCH:
            txt = await serpapi_google_search([q], max_results=min(5, k))
            # parse into lines → items
            lines = [ln for ln in (txt or "").splitlines() if ln.strip()]
            google_items: List[Dict[str, Any]] = []
            for i in range(0, len(lines), 3):
                try:
                    title = lines[i]
                    snippet = lines[i+1] if i+1 < len(lines) else ""
                    link = lines[i+2] if i+2 < len(lines) else ""
                    google_items.append({"title": title, "snippet": snippet, "link": link})
                except Exception:
                    break
            engines["google"] = google_items[:k]
        # deterministic synthetic engines
        import hashlib as _h
        def _mk(engine: str) -> List[Dict[str, Any]]:
            out = []
            for i in range(1, min(6, k) + 1):
                key = _h.sha256(f"{engine}|{q}|{i}".encode("utf-8")).hexdigest()
                out.append({"title": f"{engine} result {i}", "snippet": f"deterministic snippet {i}", "link": f"https://example.com/{engine}/{key[:16]}"})
            return out
        for eng in ("brave", "mojeek", "openalex", "gdelt"):
            engines.setdefault(eng, _mk(eng))
        fused = _rrf_fuse(engines, k=60)[:k]
        return {"name": name, "result": {"engines": list(engines.keys()), "results": fused}}
    if name == "film.run":
        # Orchestrate Film 2.0 via film2 service endpoints
        # Normalize core args
        title = (args.get("title") or "Untitled").strip()
        duration_s = int(args.get("duration_s") or _parse_duration_seconds_dynamic(args.get("duration") or args.get("duration_text"), 10))
        seed = int(args.get("seed") or _derive_seed("film", title, str(duration_s)))
        _requested_res = args.get("res")
        res = str(_requested_res or "1920x1080").lower()
        _requested_refresh = args.get("refresh")
        refresh = int(_requested_refresh or _parse_fps_dynamic(args.get("fps") or args.get("frame_rate") or "60fps", 60))
        base_fps = int(args.get("base_fps") or 30)
        # map common res synonyms
        if res in ("4k", "uhd", "2160p"): res = "3840x2160"
        if res in ("1080p", "fhd"): res = "1920x1080"
        # post defaults
        post = args.get("post") or {}
        interp = (post.get("interpolate") or {}) if isinstance(post, dict) else {}
        upscale = (post.get("upscale") or {}) if isinstance(post, dict) else {}
        if not interp.get("factor"):
            # compute factor from refresh/base_fps → prefer 2 or 4
            fac = 2 if refresh >= base_fps * 2 else 1
            if refresh >= base_fps * 4: fac = 4
            interp = {**interp, "factor": fac}
        # ensure upscale defaults include tile/overlap
        if not upscale.get("scale"):
            upscale = {**upscale, "scale": 2 if res == "3840x2160" else 1}
        if upscale.get("scale", 1) > 1:
            upscale.setdefault("tile", 512)
            upscale.setdefault("overlap", 16)
        # Aggregate style references and character images dynamically if present
        style_refs = args.get("style_refs") or []
        # If the planner passed generic images list, allow using them as style refs by default
        if not style_refs and isinstance(args.get("images"), list):
            try:
                style_refs = [it.get("url") for it in args.get("images") if isinstance(it, dict) and it.get("url")]
            except Exception:
                style_refs = []
        character_images = args.get("character_images") or []
        # Additional quality/format preferences passthrough
        quality = args.get("quality") or {}
        codec = args.get("codec") or quality.get("codec")
        container = args.get("container") or quality.get("container")
        bitrate = args.get("bitrate") or quality.get("bitrate")
        audio = args.get("audio") or {}
        audio_sr = audio.get("sr") or quality.get("audio_sr")
        lufs_target = audio.get("lufs_target") or quality.get("lufs_target")
        requested_meta = {
            "res": _requested_res,
            "refresh": _requested_refresh,
            "base_fps": base_fps,
            "post": (args.get("post") or {}),
            "style_refs": style_refs,
            "character_images": character_images,
            "duration_s": duration_s,
            "codec": codec,
            "container": container,
            "bitrate": bitrate,
            "audio_sr": audio_sr,
            "lufs_target": lufs_target,
        }
        effective_meta = {
            "res": res,
            "refresh": refresh,
            "post": {"interpolate": interp, "upscale": upscale},
            "codec": codec,
            "container": container,
            "bitrate": bitrate,
            "audio_sr": audio_sr,
            "lufs_target": lufs_target,
        }
        # Plan → Final → QC → Export
        project_id = f"prj_{seed}"
        try:
            async with httpx.AsyncClient(timeout=1200) as client:
                base_url = os.getenv("FILM2_API_URL","http://film2:8090")
                outputs_payload = args.get("outputs") or {"fps": refresh, "resolution": res, "codec": codec, "container": container, "bitrate": bitrate, "audio": {"sr": audio_sr, "lufs_target": lufs_target}}
                rules_payload = args.get("rules") or {}
                plan_payload = {"project_id": project_id, "seed": seed, "title": title, "duration_s": duration_s, "outputs": outputs_payload}
                ideas = args.get("ideas") or []
                if ideas:
                    plan_payload["ideas"] = ideas
                if style_refs:
                    plan_payload["style_refs"] = style_refs
                if character_images:
                    plan_payload["character_images"] = character_images
                await client.post(f"{base_url}/film/plan", json=plan_payload)
                br = await client.post(f"{base_url}/film/breakdown", json={"project_id": project_id, "rules": rules_payload})
                shots = []
                try:
                    js = br.json() if br.status_code == 200 else {}
                    shots = [s.get("id") for s in (js.get("shots") or []) if isinstance(s, dict) and s.get("id")]
                except Exception:
                    shots = []
                if not shots:
                    # Fallback to minimal deterministic guess
                    shots = [f"S1_SH{i+1:02d}" for i in range(max(1, min(6, int(duration_s // 3) or 1)))]
                await client.post(f"{base_url}/film/storyboard", json={"project_id": project_id, "shots": shots})
                await client.post(f"{base_url}/film/animatic", json={"project_id": project_id, "shots": shots, "outputs": {"fps": outputs_payload.get("fps", 12), "res": outputs_payload.get("resolution", "512x288")}})
                await client.post(f"{base_url}/film/final", json={"project_id": project_id, "shots": shots, "outputs": outputs_payload, "post": {"interpolate": interp, "upscale": upscale}})
                await client.post(f"{base_url}/film/qc", json={"project_id": project_id})
                exp = await client.post(f"{base_url}/film/export", json={"project_id": project_id, "post": {"interpolate": interp, "upscale": upscale}, "refresh": refresh, "requested": requested_meta, "effective": effective_meta})
                resj = exp.json() if exp.status_code == 200 else {"error": exp.text}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
        return {"name": name, "result": resj}
    if name == "source_fetch" and ALLOW_TOOL_EXECUTION:
        url = args.get("url") or ""
        if not url:
            return {"name": name, "error": "missing url"}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(url)
            ct = r.headers.get("content-type", "")
            data = r.content or b""
            import hashlib as _hl
            h = _hl.sha256(data).hexdigest()
            preview = None
            try:
                if ct.startswith("text/") or ct.find("json") >= 0:
                    txt = data.decode("utf-8", errors="ignore")
                    preview = txt[:2000]
            except Exception:
                preview = None
            return {"name": name, "result": {"status": r.status_code, "content_type": ct, "sha256": h, "preview": preview}}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "frame_interpolate":
        # Acknowledge and surface parameters; actual interpolation occurs during assembly if requested
        factor = int(args.get("factor") or 2)
        return {"name": name, "result": {"accepted": True, "factor": factor}}
    if name == "upscale":
        scale = int(args.get("scale") or 2)
        tile = int(args.get("tile") or 512)
        overlap = int(args.get("overlap") or 16)
        return {"name": name, "result": {"accepted": True, "scale": scale, "tile": tile, "overlap": overlap}}
    if name == "rag_index":
        res = await rag_index_dir(root=args.get("root", "/workspace"))
        return {"name": name, "result": res}
    if name == "rag_search":
        query = args.get("query")
        k = int(args.get("k", 8))
        if not query:
            return {"name": name, "error": "missing query"}
        res = await rag_search(query, k)
        return {"name": name, "result": res}
    # --- Multimodal external services (optional; enable via env URLs) ---
    if name == "tts_speak" and XTTS_API_URL and ALLOW_TOOL_EXECUTION:
        payload = {"text": args.get("text", ""), "voice": args.get("voice")}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(XTTS_API_URL.rstrip("/") + "/tts", json=payload)
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "asr_transcribe" and WHISPER_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(WHISPER_API_URL.rstrip("/") + "/transcribe", json={"audio_url": args.get("audio_url"), "language": args.get("language")})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "image_generate" and COMFYUI_API_URL and ALLOW_TOOL_EXECUTION:
        # expects a ComfyUI workflow graph or minimal prompt params passed through
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json=args.get("workflow") or args)
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "controlnet" and COMFYUI_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json=args.get("workflow") or args)
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "video_generate" and COMFYUI_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=1800) as client:
            r = await client.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json=args.get("workflow") or args)
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "face_embed" and FACEID_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(FACEID_API_URL.rstrip("/") + "/embed", json={"image_url": args.get("image_url")})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "music_generate" and MUSIC_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=1200) as client:
            r = await client.post(MUSIC_API_URL.rstrip("/") + "/generate", json=args)
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "vlm_analyze" and VLM_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(VLM_API_URL.rstrip("/") + "/analyze", json={"image_url": args.get("image_url"), "prompt": args.get("prompt")})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    # --- Film tools (LLM-driven, simple UI) ---
    if name == "run_python" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/run_python", json={"code": args.get("code", "")})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "write_file" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        # Guard: forbid writing banned libraries into requirements/pyproject
        p = (args.get("path") or "").lower()
        content_val = str(args.get("content", ""))
        if any(name in p for name in ("requirements.txt", "pyproject.toml")):
            banned = ("pydantic", "sqlalchemy", "pandas", "pyarrow", "fastparquet", "polars")
            if any(b in content_val.lower() for b in banned):
                return {"name": name, "error": "forbidden_library", "detail": "banned library in dependency file"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/write_file", json={"path": args.get("path"), "content": content_val})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "read_file" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/read_file", json={"path": args.get("path")})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    # MCP tool forwarding (HTTP bridge)
    if name and (name.startswith("mcp:") or args.get("mcpTool") is True) and ALLOW_TOOL_EXECUTION:
        res = await call_mcp_tool(name.replace("mcp:", "", 1), args)
        return {"name": name, "result": res}
    # Unknown tool - return as unexecuted
    return {"name": name or "unknown", "skipped": True, "reason": "unsupported tool in orchestrator"}


async def execute_tools(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for call in tool_calls[:5]:
        try:
            res = await execute_tool_call(call)
            results.append(res)
        except Exception as ex:
            results.append({"name": call.get("name", "unknown"), "error": str(ex), "traceback": traceback.format_exc()})
    return results


@app.post("/v1/chat/completions")
async def chat_completions(body: Dict[str, Any], request: Request):
    # normalize and extract attachments (images/audio/video/files) for tools
    t0 = time.time()
    # Validate body
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return JSONResponse(status_code=400, content={"error": "bad_request", "detail": "messages must be a list"})
    # Defensive bounds: messages count and total chars
    try:
        raw_msgs = body.get("messages") or []
        if len(raw_msgs) > MAX_MESSAGES:
            return JSONResponse(status_code=400, content={"error": "too_many_messages", "detail": f">{MAX_MESSAGES}"})
        total_chars = 0
        for m in raw_msgs:
            c = m.get("content")
            if isinstance(c, str):
                total_chars += len(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_chars += len(str(part.get("text") or ""))
        if total_chars > MAX_INPUT_CHARS:
            return JSONResponse(status_code=400, content={"error": "input_too_large", "detail": f">{MAX_INPUT_CHARS} chars"})
    except Exception:
        pass
    normalized_msgs, attachments = extract_attachments_from_messages(body.get("messages"))
    # prepend a system hint with attachment summary (non-invasive)
    if attachments:
        attn = json.dumps(attachments, indent=2)
        normalized_msgs = [{"role": "system", "content": f"Attachments available for tools:\n{attn}"}] + normalized_msgs
    messages = meta_prompt(normalized_msgs)

    # Determine mode and trace/run identifiers early
    last_user_text = ""
    for m in reversed(normalized_msgs):
        if (m.get("role") == "user") and isinstance(m.get("content"), str) and m.get("content").strip():
            last_user_text = m.get("content").strip(); break
    mode = "film" if _detect_video_intent(last_user_text) else "general"
    # generate a deterministic trace id from messages
    try:
        msgs_for_seed = json.dumps([{"role": m.get("role"), "content": m.get("content")} for m in normalized_msgs], ensure_ascii=False, separators=(",", ":"))
    except Exception:
        msgs_for_seed = ""
    provided_seed = None
    try:
        if isinstance(body.get("seed"), (int, float)):
            provided_seed = int(body.get("seed"))
    except Exception:
        provided_seed = None
    master_seed = provided_seed if provided_seed is not None else _derive_seed("chat", msgs_for_seed)
    import hashlib as _hl
    trace_id = "tt_" + _hl.sha256(msgs_for_seed.encode("utf-8")).hexdigest()[:16]
    # Acquire per-trace lock and record start event
    _lock_token = None
    try:
        _lock_token = _acquire_lock(STATE_DIR, trace_id, timeout_s=10)
    except Exception:
        _lock_token = None
    try:
        _append_event(STATE_DIR, trace_id, "start", {"seed": int(master_seed), "mode": mode})
    except Exception:
        pass

    # ICW pack (always-on, unless ICW_DISABLE=1) — inline; record pack_hash for traces
    pack_hash = None
    try:
        seed_icw = _derive_seed("icw", msgs_for_seed)
        if not ICW_DISABLE:
            icw = _icw_pack(normalized_msgs, seed_icw, budget_tokens=3500)
            pack_text = icw.get("pack") or ""
            if isinstance(pack_text, str) and pack_text.strip():
                messages = [{"role": "system", "content": f"ICW PACK (hash tracked):\n{pack_text[:12000]}"}] + messages
            pack_hash = icw.get("hash")
            run_id = await _db_insert_run(trace_id=trace_id, mode=mode, seed=master_seed, pack_hash=pack_hash, request_json=body)
            await _db_insert_icw_log(run_id=run_id, pack_hash=pack_hash or None, budget_tokens=int(icw.get("budget_tokens") or 0), scores_json=icw.get("scores_summary") or {})
        else:
            run_id = await _db_insert_run(trace_id=trace_id, mode=mode, seed=master_seed, pack_hash=None, request_json=body)
    except Exception:
        pack_hash = None
        run_id = await _db_insert_run(trace_id=trace_id, mode=mode, seed=master_seed, pack_hash=None, request_json=body)

    # If client supplies tool results (role=tool) we include them verbatim for the planner/executors

    # Optional Windowed Solver path (capless sliding window + CONT/HALT)
    route_mode = os.getenv("ICW_MODE", "windowed").lower()
    if route_mode == "windowed":
        # Build minimal state
        anchor_msgs = []
        for m in normalized_msgs[-6:]:
            if isinstance(m.get("content"), str) and m.get("content").strip():
                role = m.get("role")
                anchor_msgs.append(f"{role}: {m.get('content')}")
        anchor_text = "\n".join(anchor_msgs[-4:])
        try:
            pack_text = icw.get("pack") if (not ICW_DISABLE) else ""
        except Exception:
            pack_text = ""
        state = {
            "anchor_text": anchor_text,
            "entities": [],
            "candidates": [t for t in [pack_text] if isinstance(t, str) and t.strip()],
            "cid": trace_id,
        }
        # Sync Ollama provider
        class _SyncOllamaProvider:
            def __init__(self, base_url: str, model_id: str, num_ctx: int, temperature: float):
                self.base_url = base_url.rstrip("/")
                self.model_id = model_id
                self.num_ctx = num_ctx
                self.temperature = temperature
            def chat(self, prompt: str, max_tokens: int) -> SimpleNamespace:
                payload = {
                    "model": self.model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": self.num_ctx,
                        "temperature": self.temperature,
                        "keep_alive": "24h",
                        "num_predict": max(1, int(max_tokens or 900)),
                    },
                }
                try:
                    r = requests.post(self.base_url + "/api/generate", json=payload, timeout=60)
                    r.raise_for_status()
                    js = r.json()
                    return SimpleNamespace(text=str(js.get("response", "")), model_name=self.model_id)
                except Exception as ex:
                    return SimpleNamespace(text=f"", model_name=self.model_id)
        ctx_limit = DEFAULT_NUM_CTX
        step_out = int(os.getenv("ICW_STEP_TOKENS", "900") or 900)
        provider = _SyncOllamaProvider(QWEN_BASE_URL, QWEN_MODEL_ID, ctx_limit, body.get("temperature") or DEFAULT_TEMPERATURE)
        def _progress(step: int, kind: str, info: Dict[str, Any] | None = None):
            try:
                _append_event(STATE_DIR, trace_id, f"window_{kind}", {"step": int(step), **(info or {})})
            except Exception:
                pass
        result = windowed_solve(
            request={"content": last_user_text},
            global_state=state,
            model=provider,
            model_ctx_limit_tokens=ctx_limit,
            step_out_tokens=step_out,
            progress=_progress,
        )
        # Normalize envelopes (optional log material)
        step_envs = [normalize_to_envelope(p) for p in (result.partials or [])]
        final_env = stitch_merge_envelopes(step_envs)
        final_oai = stitch_openai_final(result.partials, model_name=provider.model_id)
        # Optional ablation
        try:
            do_ablate = os.getenv("ABLATE", "on").lower() == "on"
            do_export = os.getenv("ABLATE_EXPORT", "on").lower() == "on"
            scope = os.getenv("ABLATE_SCOPE", "auto")
            if do_ablate and isinstance(final_env, dict):
                abl = ablate_env(final_env, scope_hint=(scope if scope != "auto" else "chat"))
                final_env["ablated"] = abl
                if do_export:
                    try:
                        outdir = os.path.join(UPLOAD_DIR, "ablation", trace_id)
                        facts_path = ablate_write_facts(abl, trace_id, outdir)
                        # Record in manifest
                        try:
                            mdir = os.path.join(UPLOAD_DIR, "manifests", trace_id)
                            mpath = os.path.join(mdir, "manifest.json")
                            existing = {}
                            if os.path.exists(mpath):
                                try:
                                    with open(mpath, "r", encoding="utf-8") as _mf:
                                        existing = json.load(_mf)
                                except Exception:
                                    existing = {}
                            sid = _step_id()
                            _manifest_add_row(existing, facts_path, sid)
                            _manifest_write(mdir, existing)
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass
        # Build response wrapper compatible with our existing format
        final_text = (final_oai.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
        usage = estimate_usage(messages, final_text)
        response = {
            "id": final_oai.get("id", "orc-1"),
            "object": "chat.completion",
            "model": final_oai.get("model") or f"{QWEN_MODEL_ID}",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": final_text}}],
            "usage": usage,
            "seed": master_seed,
        }
        if isinstance(final_env, dict):
            response["envelope"] = final_env
        # Persist response & metrics
        await _db_update_run_response(run_id, response, usage)
        # Record finalization for state tracking
        try:
            _append_event(STATE_DIR, trace_id, "halt", {"kind": "windowed", "chars": len(final_text)})
        except Exception:
            pass
        if body.get("stream"):
            async def _stream_once():
                chunk = json.dumps({"id": response["id"], "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": final_text}, "finish_reason": None}]})
                yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            try:
                if _lock_token:
                    _release_lock(STATE_DIR, trace_id)
            except Exception:
                pass
            return StreamingResponse(_stream_once(), media_type="text/event-stream")
        try:
            if _lock_token:
                _release_lock(STATE_DIR, trace_id)
        except Exception:
            pass
        return JSONResponse(content=response)

    # Optional self-ask-with-search augmentation
    if ENABLE_WEBSEARCH and SERPAPI_API_KEY:
        queries = await propose_search_queries(messages)
        if queries:
            snippets = await serpapi_google_search(queries, max_results=5)
            if snippets:
                messages = [
                    {"role": "system", "content": "Web search results follow. Use them only if relevant."},
                    {"role": "system", "content": snippets},
                ] + messages

    # 1) Planner proposes plan + tool calls
    plan_text, tool_calls = await planner_produce_plan(messages, body.get("tools"), body.get("temperature") or DEFAULT_TEMPERATURE)
    # Deterministic router: if intent is recognized, override planner with a direct tool call
    try:
        decision = route_for_request({"messages": normalized_msgs})
    except Exception:
        decision = None
    if decision and getattr(decision, "kind", None) == "tool" and getattr(decision, "tool", None):
        tool_calls = [{"name": decision.tool, "arguments": (decision.args or {})}]
    # Surface attachments in tool arguments so downstream jobs can use user media
    if tool_calls and attachments:
        enriched: List[Dict[str, Any]] = []
        for tc in tool_calls:
            args = tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"_raw": args}
            # Attachments by type for convenience
            imgs = [a for a in attachments if a.get("type") == "image"]
            auds = [a for a in attachments if a.get("type") == "audio"]
            vids = [a for a in attachments if a.get("type") == "video"]
            files = [a for a in attachments if a.get("type") == "file"]
            if imgs:
                args.setdefault("images", imgs)
            if auds:
                args.setdefault("audio", auds)
            if vids:
                args.setdefault("video", vids)
            if files:
                args.setdefault("files", files)
            tc = {**tc, "arguments": args}
            enriched.append(tc)
        tool_calls = enriched
    # tool_choice=required compatibility: force at least one tool_call if tools are provided
    if (body.get("tool_choice") == "required") and (not tool_calls):
        # Choose a sensible default tool with minimal required params
        builtins = get_builtin_tools_schema()
        chosen = None
        fewest_required = 1e9
        for t in builtins:
            fn = (t.get("function") or {})
            name = fn.get("name")
            req = ((fn.get("parameters") or {}).get("required") or [])
            if name and len(req) < fewest_required:
                chosen = name
                fewest_required = len(req)
        if chosen:
            tool_calls = [{"name": chosen, "arguments": {}}]

    # If no tool_calls were proposed but tools are available, nudge Planner by ensuring tools context is always present
    # (No keyword heuristics; semantic mapping is handled by the Planner instructions above.)

    # If tool semantics are client-driven, return tool_calls instead of executing
    if tool_calls and not AUTO_EXECUTE_TOOLS:
        tool_calls_openai = to_openai_tool_calls(tool_calls)
        response = {
            "id": "orc-1",
            "object": "chat.completion",
            "model": f"committee:{QWEN_MODEL_ID}+{GPTOSS_MODEL_ID}",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_openai,
                    },
                }
            ],
        }
        if body.get("stream"):
            async def _stream_once():
                chunk = json.dumps({"id": response["id"], "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": tool_calls_openai}, "finish_reason": None}]})
                yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            try:
                if _lock_token:
                    _release_lock(STATE_DIR, trace_id)
            except Exception:
                pass
            return StreamingResponse(_stream_once(), media_type="text/event-stream")
        # include usage estimate even in tool_calls path (no completion tokens yet)
        response["usage"] = estimate_usage(messages, "")
        response["seed"] = master_seed
        try:
            if _lock_token:
                _release_lock(STATE_DIR, trace_id)
        except Exception:
            pass
        return JSONResponse(content=response)

    # 2) Optionally execute tools
    tool_results: List[Dict[str, Any]] = []
    tool_exec_meta: List[Dict[str, Any]] = []
    # Artifacts ledger (lazy-open)
    _ledger_shard = None
    _ledger_root = os.path.join(UPLOAD_DIR, "artifacts", "runs", trace_id)
    _ledger_name = "ledger"
    # Enforce tool call budget
    if tool_calls and len(tool_calls) > MAX_TOOL_CALLS:
        return JSONResponse(status_code=429, content={"error": "tool_budget_exceeded", "detail": f">{MAX_TOOL_CALLS} tool calls"})
    if tool_calls:
        for tc in tool_calls[:5]:
            try:
                n = tc.get("name") or "tool"
                args = tc.get("arguments") or {}
                if n == "research.run":
                    try:
                        if isinstance(args, dict):
                            args.setdefault("cid", trace_id)
                    except Exception:
                        pass
                seed_tool = det_seed_tool(n, trace_id, master_seed)
                tstart = time.time()
                tr = await execute_tool_call(tc)
                duration_ms = int(round((time.time() - tstart) * 1000))
                # Clamp potentially large string outputs in tool result
                try:
                    if isinstance(tr, dict) and isinstance(tr.get("result"), dict):
                        resd = tr.get("result")
                        prev = resd.get("preview")
                        if isinstance(prev, str) and len(prev) > MAX_TOOL_OUTPUT_CHARS:
                            resd["preview"] = prev[:MAX_TOOL_OUTPUT_CHARS] + "\n... [truncated]"
                except Exception:
                    pass
                tool_results.append(tr)
                # persist to DB
                try:
                    await _db_insert_tool_call(run_id, n, seed_tool, args if isinstance(args, dict) else {"_raw": args}, tr if isinstance(tr, dict) else {}, duration_ms)
                except Exception:
                    pass
                # append checkpoint event
                try:
                    _append_event(STATE_DIR, trace_id, "tool_call", {"name": n, "duration_ms": int(duration_ms), "ok": (not (isinstance(tr, dict) and tr.get("error")))})
                except Exception:
                    pass
                # Append a compact ledger row for this tool call
                try:
                    if _ledger_shard is None:
                        _ledger_shard = _art_open_shard(_ledger_root, _ledger_name, ARTIFACT_SHARD_BYTES)
                    row = {"tool": n, "duration_ms": int(duration_ms)}
                    if isinstance(args, dict):
                        row["args_keys"] = list(args.keys())[:10]
                    if isinstance(tr, dict):
                        if tr.get("error"):
                            row["error"] = str(tr.get("error"))[:200]
                        res = tr.get("result") if isinstance(tr.get("result"), dict) else {}
                        if res:
                            for k in ("master_uri", "hash", "package_uri"):
                                v = res.get(k)
                                if isinstance(v, str) and v:
                                    row[k] = v
                    _ledger_shard = _art_append_jsonl(_ledger_shard, row)
                except Exception:
                    pass
                # extract artifacts minimally for trace policy
                artifacts = {}
                if isinstance(tr, dict):
                    res = tr.get("result") if isinstance(tr.get("result"), dict) else {}
                    if res:
                        if res.get("master_uri") and res.get("hash"):
                            artifacts["master"] = {"uri": res.get("master_uri"), "hash": res.get("hash")}
                        for key in ("edl", "nodes", "qc_report"):
                            if isinstance(res.get(key), dict) and (res.get(key).get("uri") or res.get(key).get("hash")):
                                artifacts[key] = {k: v for k, v in res.get(key).items() if k in ("uri", "hash")}
                tool_exec_meta.append({"name": n, "args": args if isinstance(args, dict) else {"_raw": args}, "seed": seed_tool, "duration_ms": duration_ms, "artifacts": artifacts})
            except Exception as ex:
                tool_results.append({"name": tc.get("name", "tool"), "error": str(ex)})
                tool_exec_meta.append({"name": tc.get("name", "tool"), "args": tc.get("arguments") or {}, "seed": _derive_seed("tool", tc.get("name", "tool"), trace_id), "duration_ms": 0, "artifacts": {}})
        if tool_results:
            messages = [{"role": "system", "content": "Tool results:\n" + json.dumps(tool_results, indent=2)}] + messages

    # 3) Executors respond independently using plan + evidence
    evidence_blocks: List[Dict[str, Any]] = []
    if plan_text:
        evidence_blocks.append({"role": "system", "content": f"Planner plan:\n{plan_text}"})
    if tool_results:
        evidence_blocks.append({"role": "system", "content": "Tool results:\n" + json.dumps(tool_results, indent=2)})
    if attachments:
        evidence_blocks.append({"role": "system", "content": "User attachments (for tools):\n" + json.dumps(attachments, indent=2)})
    # If tool results include errors, nudge executors to include brief, on-topic suggestions
    exec_messages = evidence_blocks + messages
    exec_messages_current = exec_messages
    if any(isinstance(r, dict) and r.get("error") for r in tool_results or []):
        exec_messages = exec_messages + [{"role": "system", "content": (
            "If the tool results above contain errors that block the user's goal, include a short 'Suggestions' section (max 2 bullets) with specific, on-topic fixes (e.g., missing parameter defaults, retry guidance). Keep it brief and avoid scope creep."
        )}]

    qwen_payload = build_ollama_payload(
        messages=exec_messages, model=QWEN_MODEL_ID, num_ctx=DEFAULT_NUM_CTX, temperature=body.get("temperature") or DEFAULT_TEMPERATURE
    )
    gptoss_payload = build_ollama_payload(
        messages=exec_messages, model=GPTOSS_MODEL_ID, num_ctx=DEFAULT_NUM_CTX, temperature=body.get("temperature") or DEFAULT_TEMPERATURE
    )

    qwen_task = asyncio.create_task(call_ollama(QWEN_BASE_URL, qwen_payload))
    gptoss_task = asyncio.create_task(call_ollama(GPTOSS_BASE_URL, gptoss_payload))
    qwen_result, gptoss_result = await asyncio.gather(qwen_task, gptoss_task)
    # Fast-fail if either backend errored
    if qwen_result.get("error") or gptoss_result.get("error"):
        detail = {
            "qwen": {k: v for k, v in qwen_result.items() if k in ("error", "_base_url")},
            "gptoss": {k: v for k, v in gptoss_result.items() if k in ("error", "_base_url")},
        }
        try:
            if _lock_token:
                _release_lock(STATE_DIR, trace_id)
        except Exception:
            pass
        return JSONResponse(status_code=502, content={"error": "backend_failed", "detail": detail})

    qwen_text = qwen_result.get("response", "")
    gptoss_text = gptoss_result.get("response", "")
    # Preserve the first-pass model outputs for robust fallback in final composition
    orig_qwen_text = qwen_text
    orig_gptoss_text = gptoss_text

    # Safety correction: if executors refuse despite available tools, trigger semantic tool path
    # Detect generic refusal; if tools can fulfill and no tools were executed yet, synthesize a best-match tool call
    refusal_markers = ("can't", "cannot", "unable", "i can't", "i can't", "i cannot")
    refused = any(tok in (qwen_text.lower() + "\n" + gptoss_text.lower()) for tok in refusal_markers)
    if refused and not tool_results:
        # Safe semantic forcing: only for tools we can call without hidden context (no film_id requirements)
        # 1) Identify latest user text
        last_user = ""
        for m in reversed(messages):
            if m.role == "user" and isinstance(m.content, str) and m.content.strip():
                last_user = m.content.strip()
                break
        # 2) Merge tools and score semantic overlap
        merged_tools = merge_tool_schemas(body.get("tools"))
        allowed_tools: List[Tuple[str, Dict[str, Any]]] = []
        for t in merged_tools:
            fn = (t.get("function") or {})
            name = fn.get("name") or t.get("name")
            params = (fn.get("parameters") or {})
            required = (params.get("required") or [])
            # Skip tools that require film_id or other unavailable hard requirements
            if any(req in ("film_id",) for req in required):
                continue
            if name:
                allowed_tools.append((name, fn))
        # 3) Pick best match
        best_name = None
        best_score = -1
        ux_tokens = set([w for w in (last_user or "").lower().replace("\n", " ").split(" ") if w])
        for name, fn in allowed_tools:
            desc = (fn.get("description") or "").lower()
            corpus_tokens = set([w for w in (name.replace("_"," ") + " " + desc).split(" ") if w])
            score = len(ux_tokens & corpus_tokens)
            if score > best_score:
                best_name = name
                best_score = score
        MIN_SEMANTIC_SCORE = 3
        forced_calls: List[Dict[str, Any]] = []
        if best_name and best_score >= MIN_SEMANTIC_SCORE:
            # 4) Build minimal arguments by tool
            def _minimal_args(tool_name: str, text: str) -> Dict[str, Any]:
                if tool_name == "make_movie":
                    prefs = _derive_movie_prefs_from_text(text)
                    return {**prefs, "synopsis": text}
                if tool_name == "film_create":
                    prefs = _derive_movie_prefs_from_text(text)
                    return {"title": "Untitled", "synopsis": text, "metadata": prefs}
                if tool_name == "rag_search":
                    return {"query": text, "k": 8}
                if tool_name == "tts_speak":
                    return {"text": text}
                if tool_name in ("image_generate", "video_generate", "controlnet"):
                    return {"prompt": text}
                return {}
            forced_calls = [{"name": best_name, "arguments": _minimal_args(best_name, last_user)}]
        # 5) Execute if any, then include results for synthesis context
        if forced_calls:
            try:
                tool_results = await execute_tools(forced_calls)
                evidence_blocks = [{"role": "system", "content": "Tool results:\n" + json.dumps(tool_results, indent=2)}]
                exec_messages2 = evidence_blocks + messages
                qwen_payload2 = build_ollama_payload(messages=exec_messages2, model=QWEN_MODEL_ID, num_ctx=DEFAULT_NUM_CTX, temperature=body.get("temperature") or DEFAULT_TEMPERATURE)
                gptoss_payload2 = build_ollama_payload(messages=exec_messages2, model=GPTOSS_MODEL_ID, num_ctx=DEFAULT_NUM_CTX, temperature=body.get("temperature") or DEFAULT_TEMPERATURE)
                qwen_res2 = await call_ollama(QWEN_BASE_URL, qwen_payload2)
                gptoss_res2 = await call_ollama(GPTOSS_BASE_URL, gptoss_payload2)
                # Do not discard the original answers; append improved content if any
                new_q = qwen_res2.get("response", "")
                new_g = gptoss_res2.get("response", "")
                if isinstance(new_q, str) and new_q.strip():
                    qwen_text = (orig_qwen_text or qwen_text or "") + ("\n\n" + new_q)
                if isinstance(new_g, str) and new_g.strip():
                    gptoss_text = (orig_gptoss_text or gptoss_text or "") + ("\n\n" + new_g)
                exec_messages_current = exec_messages2
            except Exception as ex:
                tool_results = tool_results or [{"name": best_name or "unknown", "error": str(ex)}]

    # If response still looks like a refusal, synthesize a constructive message using tool_results
    final_refusal = any(tok in (qwen_text.lower() + "\n" + gptoss_text.lower()) for tok in refusal_markers)
    if final_refusal:
        # Try to extract a film_id or any success info from tool results
        film_id = None
        errors: List[str] = []
        for tr in (tool_results or []):
            if isinstance(tr, dict):
                rid = ((tr.get("result") or {}).get("film_id"))
                if rid and not film_id:
                    film_id = rid
                if tr.get("error"):
                    errors.append(str(tr.get("error")))
        details = []
        if film_id:
            details.append(f"Film ID: {film_id}")
        if errors:
            details.append("Errors: " + "; ".join(errors)[:800])
        summary = "\n".join(details)
        affirmative = (
            "Initiated tool-based generation flow. "
            + ("\n" + summary if summary else "")
            + "\nUse film_status to track progress, and jobs endpoints for live status."
        )
        # Do not modify model texts here; the final composer will only append status

    # 4) Optional brief debate (cross-critique)
    if ENABLE_DEBATE and MAX_DEBATE_TURNS > 0:
        critique_prompt = (
            "Critique the other model's answer. Identify mistakes, missing considerations, or improvements. "
            "Return a concise bullet list."
        )
        qwen_critique_msg = exec_messages + [{"role": "user", "content": critique_prompt + f"\nOther answer:\n{gptoss_text}"}]
        gptoss_critique_msg = exec_messages + [{"role": "user", "content": critique_prompt + f"\nOther answer:\n{qwen_text}"}]
        qwen_crit_payload = build_ollama_payload(qwen_critique_msg, QWEN_MODEL_ID, DEFAULT_NUM_CTX, body.temperature or DEFAULT_TEMPERATURE)
        gptoss_crit_payload = build_ollama_payload(gptoss_critique_msg, GPTOSS_MODEL_ID, DEFAULT_NUM_CTX, body.temperature or DEFAULT_TEMPERATURE)
        qcrit_task = asyncio.create_task(call_ollama(QWEN_BASE_URL, qwen_crit_payload))
        gcrit_task = asyncio.create_task(call_ollama(GPTOSS_BASE_URL, gptoss_crit_payload))
        qcrit_res, gcrit_res = await asyncio.gather(qcrit_task, gcrit_task)
        qcrit_text = qcrit_res.get("response", "")
        gcrit_text = gcrit_res.get("response", "")
        exec_messages = exec_messages + [
            {"role": "system", "content": "Cross-critique from Qwen:\n" + qcrit_text},
            {"role": "system", "content": "Cross-critique from GPT-OSS:\n" + gcrit_text},
        ]

    # 5) Final synthesis by Planner
    final_request = exec_messages_current + [{"role": "user", "content": (
        "Produce the final, corrected answer, incorporating critiques and evidence. "
        "Be unambiguous, include runnable code when requested, and prefer specific citations to tool results."
    )}]

    planner_id = QWEN_MODEL_ID if PLANNER_MODEL.lower() == "qwen" else GPTOSS_MODEL_ID
    planner_base = QWEN_BASE_URL if PLANNER_MODEL.lower() == "qwen" else GPTOSS_BASE_URL
    synth_payload = build_ollama_payload(final_request, planner_id, DEFAULT_NUM_CTX, body.temperature or DEFAULT_TEMPERATURE)
    synth_result = await call_ollama(planner_base, synth_payload)
    final_text = synth_result.get("response", "") or qwen_text or gptoss_text

    if body.get("stream"):
        # Precompute usage and planned stage events for trace
        usage_stream = estimate_usage(messages, final_text)
        planned_events: List[Dict[str, Any]] = []
        planned_events.append({"stage": "plan", "ok": True})
        has_film = any(isinstance(tc, dict) and str(tc.get("name", "")).startswith("film_") for tc in (tool_calls or []))
        if has_film:
            for st in ("breakdown", "storyboard", "animatic"):
                planned_events.append({"stage": st})
        for tc in (tool_calls or [])[:20]:
            try:
                name = tc.get("name") or tc.get("function", {}).get("name")
                args = tc.get("arguments") or tc.get("args") or {}
                if name == "film_add_scene":
                    planned_events.append({"stage": "final", "shot_id": args.get("index_num") or args.get("scene_id")})
                if name == "frame_interpolate":
                    planned_events.append({"stage": "post", "op": "frame_interpolate", "factor": args.get("factor")})
                if name == "upscale":
                    planned_events.append({"stage": "post", "op": "upscale", "scale": args.get("scale")})
                if name == "film_compile":
                    planned_events.append({"stage": "export"})
            except Exception:
                continue
        if isinstance(plan_text, str) and ("qc" in plan_text.lower() or "quality" in plan_text.lower()):
            planned_events.append({"stage": "qc"})

        # Fire-and-forget teacher trace for streaming path as well
        try:
            req_dict = dict(body)
            try:
                msgs_for_seed = json.dumps(req_dict.get("messages", []), ensure_ascii=False, separators=(",", ":"))
            except Exception:
                msgs_for_seed = ""
            provided_seed3 = None
            try:
                if isinstance(body.get("seed"), (int, float)):
                    provided_seed3 = int(body.get("seed"))
            except Exception:
                provided_seed3 = None
            master_seed = provided_seed3 if provided_seed3 is not None else _derive_seed("chat", msgs_for_seed)
            seed_router = det_seed_router(trace_id, master_seed)
            label_cfg = None
            try:
                label_cfg = (WRAPPER_CONFIG.get("teacher") or {}).get("default_label")
            except Exception:
                label_cfg = None
            # Normalize tool_calls shape for teacher (use args, not arguments)
            _tc = tool_exec_meta or []
            trace_payload_stream = {
                "label": label_cfg or "exp_default",
                "seed": master_seed,
                "request": {"messages": req_dict.get("messages", []), "tools_allowed": [t.get("function", {}).get("name") for t in (body.get("tools") or []) if isinstance(t, dict)]},
                "context": ({"pack_hash": pack_hash} if ("pack_hash" in locals() and pack_hash) else {}),
                "routing": {"planner_model": planner_id, "executors": [QWEN_MODEL_ID, GPTOSS_MODEL_ID], "seed_router": seed_router},
                "tool_calls": _tc,
                "response": {"text": (final_text or "")[:4000]},
                "metrics": usage_stream,
                "env": {"public_base_url": PUBLIC_BASE_URL, "config_hash": WRAPPER_CONFIG_HASH},
                "privacy": {"vault_refs": 0, "secrets_in": 0, "secrets_out": 0},
                "events": planned_events[:64],
            }
            async def _send_trace_stream():
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        await client.post(TEACHER_API_URL.rstrip("/") + "/teacher/trace.append", json=trace_payload_stream)
                except Exception:
                    return
            asyncio.create_task(_send_trace_stream())
        except Exception:
            pass

        async def _stream_with_stages(text: str):
            # Open the stream with assistant role
            now = int(time.time())
            model_id = f"committee:{QWEN_MODEL_ID}+{GPTOSS_MODEL_ID}"
            head = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": now, "model": model_id, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            yield f"data: {head}\n\n"
            # Progress events as content JSON lines to match example
            try:
                # plan
                evt = {"stage": "plan", "ok": True}
                yield f"data: {json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {"content": json.dumps(evt)} , "finish_reason": None}]})}\n\n"
                # breakdown/storyboard/animatic heuristics: if any film tool present, emit these scaffolding stages
                has_film = any(isinstance(tc, dict) and str(tc.get("name", "")).startswith("film_") for tc in (tool_calls or []))
                if has_film:
                    for st in ("breakdown", "storyboard", "animatic"):
                        ev = {"stage": st}
                        yield f"data: {json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {"content": json.dumps(ev)} , "finish_reason": None}]})}\n\n"
                # Emit per-tool mapped events
                for tc in (tool_calls or [])[:20]:
                    try:
                        name = tc.get("name") or tc.get("function", {}).get("name")
                        args = tc.get("arguments") or tc.get("args") or {}
                        if name == "film_add_scene":
                            ev = {"stage": "final", "shot_id": args.get("index_num") or args.get("scene_id")}
                            yield f"data: {json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {"content": json.dumps(ev)} , "finish_reason": None}]})}\n\n"
                        if name == "frame_interpolate":
                            ev = {"stage": "post", "op": "frame_interpolate", "factor": args.get("factor")}
                            yield f"data: {json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {"content": json.dumps(ev)} , "finish_reason": None}]})}\n\n"
                        if name == "upscale":
                            ev = {"stage": "post", "op": "upscale", "scale": args.get("scale")}
                            yield f"data: {json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {"content": json.dumps(ev)} , "finish_reason": None}]})}\n\n"
                        if name == "film_compile":
                            ev = {"stage": "export"}
                            yield f"data: {json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {"content": json.dumps(ev)} , "finish_reason": None}]})}\n\n"
                    except Exception:
                        continue
                # Optionally signal qc if present in plan text
                if isinstance(plan_text, str) and ("qc" in plan_text.lower() or "quality" in plan_text.lower()):
                    ev = {"stage": "qc"}
                    yield f"data: {json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {"content": json.dumps(ev)} , "finish_reason": None}]})}\n\n"
            except Exception:
                pass
            # Stream final content
            if STREAM_CHUNK_SIZE_CHARS and STREAM_CHUNK_SIZE_CHARS > 0:
                size = max(1, STREAM_CHUNK_SIZE_CHARS)
                for i in range(0, len(text), size):
                    piece = text[i : i + size]
                    chunk = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                    yield f"data: {chunk}\n\n"
                    if STREAM_CHUNK_INTERVAL_MS > 0:
                        await asyncio.sleep(STREAM_CHUNK_INTERVAL_MS / 1000.0)
            else:
                content_chunk = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
                yield f"data: {content_chunk}\n\n"
            done = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model_id, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield f"data: {done}\n\n"
            yield "data: [DONE]\n\n"
        try:
            _append_event(STATE_DIR, trace_id, "halt", {"kind": "committee", "chars": len(final_text)})
        except Exception:
            pass
        try:
            if _lock_token:
                _release_lock(STATE_DIR, trace_id)
        except Exception:
            pass
        return StreamingResponse(_stream_with_stages(final_text), media_type="text/event-stream")

    # Merge exact usages if available, else approximate
    usage = merge_usages([
        qwen_result.get("_usage"),
        gptoss_result.get("_usage"),
        synth_result.get("_usage"),
    ])
    if usage["total_tokens"] == 0:
        usage = estimate_usage(messages, final_text)
    wall_ms = int(round((time.time() - t0) * 1000))
    usage_with_wall = dict(usage)
    usage_with_wall["wall_ms"] = wall_ms

    # Ensure clean markdown content: collapse excessive whitespace but keep newlines
    cleaned = final_text.replace('\r\n', '\n')
    # If the planner synthesis is empty or trivially short, fall back to merged model answers as the main body
    def _is_trivial(s: str) -> bool:
        return not s.strip() or len(s.strip()) < 8
    if _is_trivial(cleaned):
        merged_main = []
        if (qwen_text or '').strip():
            merged_main.append("### Qwen\n" + qwen_text.strip())
        if (gptoss_text or '').strip():
            merged_main.append("### GPT‑OSS\n" + gptoss_text.strip())
        if merged_main:
            cleaned = "\n\n".join(merged_main)
    # If tools/plan/critique exist, wrap them into a hidden metadata block and present a concise final answer
    meta_sections: List[str] = []
    if plan_text:
        meta_sections.append("Plan:\n" + plan_text)
    if tool_results:
        try:
            meta_sections.append("Tool Results:\n" + json.dumps(tool_results, indent=2))
        except Exception:
            pass
    # Instead of verbose plan/critique footers, append a minimal Tool Results summary (IDs/errors) only
    tool_summary_lines: List[str] = []
    tool_tracebacks: List[str] = []
    if tool_results:
        try:
            film_id = None
            errors: List[str] = []
            job_ids: List[str] = []
            prompt_ids: List[str] = []
            for tr in (tool_results or []):
                if isinstance(tr, dict):
                    # Safely normalize result to a dict before any .get
                    res_field = tr.get("result")
                    if isinstance(res_field, str):
                        try:
                            res_field = JSONParser().parse(res_field, {"film_id": str, "job_id": str, "prompt_id": str, "created": [ {"result": dict, "job_id": str, "prompt_id": str} ]})
                        except Exception:
                            res_field = {}
                    if not isinstance(res_field, dict):
                        res_field = {}
                    if not film_id:
                        film_id = res_field.get("film_id")
                    if tr.get("error"):
                        errors.append(str(tr.get("error")))
                    tb = tr.get("traceback")
                    if isinstance(tb, str) and tb.strip():
                        tool_tracebacks.append(tb)
                    # direct job_id/prompt_id
                    res = res_field
                    jid = res.get("job_id") or tr.get("job_id")
                    pid = res.get("prompt_id") or tr.get("prompt_id")
                    if isinstance(jid, str):
                        job_ids.append(jid)
                    if isinstance(pid, str):
                        prompt_ids.append(pid)
                    # make_movie created list
                    created = res.get("created") if isinstance(res, dict) else None
                    if isinstance(created, list):
                        for it in created:
                            if isinstance(it, dict):
                                r2 = it.get("result") or {}
                                j2 = r2.get("job_id") or it.get("job_id")
                                p2 = r2.get("prompt_id") or it.get("prompt_id")
                                if isinstance(j2, str):
                                    job_ids.append(j2)
                                if isinstance(p2, str):
                                    prompt_ids.append(p2)
            if film_id:
                tool_summary_lines.append(f"- **film_id**: `{film_id}`")
            if job_ids:
                juniq = list(dict.fromkeys([j for j in job_ids if j]))
                if juniq:
                    tool_summary_lines.append("- **jobs**: " + ", ".join([f"`{j}`" for j in juniq[:20]]))
            if prompt_ids:
                puniq = list(dict.fromkeys([p for p in prompt_ids if p]))
                if puniq:
                    tool_summary_lines.append("- **prompts**: " + ", ".join([f"`{p}`" for p in puniq[:20]]))
            if errors:
                tool_summary_lines.append("- **errors**: " + "; ".join(errors)[:800])
        except Exception:
            pass
    footer = ("\n\n### Tool Results\n" + "\n".join(tool_summary_lines)) if tool_summary_lines else ""
    if tool_tracebacks:
        # Include full tracebacks verbatim (truncated to a safe length per item)
        def _tb(s: str) -> str:
            return s if len(s) <= 16000 else (s[:16000] + "\n... [traceback truncated]")
        footer += "\n\n### Debug — Tracebacks\n" + "\n\n".join([f"```\n{_tb(t)}\n```" for t in tool_tracebacks])
    # Decide emptiness/refusal based on the main body only, not the footer
    main_only = cleaned
    text_lower = (main_only or "").lower()
    refusal_markers = ("can't", "cannot", "unable", "not feasible", "can't create", "cannot create", "won't be able", "won't be able")
    looks_empty = (not str(main_only or "").strip())
    looks_refusal = any(tok in text_lower for tok in refusal_markers)
    display_content = f"{cleaned}{footer}"
    # Provide an appendix with raw model answers to avoid any perception of truncation
    try:
        raw_q = (qwen_text or "").strip()
        raw_g = (gptoss_text or "").strip()
        appendix_parts: List[str] = []
        def _shorten(s: str, max_len: int = 12000) -> str:
            return s if len(s) <= max_len else (s[:max_len] + "\n... [truncated]")
        if raw_q or raw_g:
            appendix_parts.append("\n\n### Appendix — Model Answers")
        if raw_q:
            appendix_parts.append("\n\n#### Qwen\n" + _shorten(raw_q))
        if raw_g:
            appendix_parts.append("\n\n#### GPT‑OSS\n" + _shorten(raw_g))
        if appendix_parts:
            display_content += "".join(appendix_parts)
    except Exception:
        pass
    # Evaluate fallback after footer creation, but using main-only flags
    if looks_empty or looks_refusal:
        # Build a constructive status from tool_results
        film_id = None
        errors = []
        if isinstance(tool_results, list):
            for tr in tool_results:
                if isinstance(tr, dict):
                    # Safely normalize result to a dict before any .get
                    res_field = tr.get("result")
                    if isinstance(res_field, str):
                        try:
                            res_field = JSONParser().parse(res_field, {"film_id": str})
                        except Exception:
                            res_field = {}
                    if not isinstance(res_field, dict):
                        res_field = {}
                    if not film_id:
                        film_id = res_field.get("film_id")
                    if tr.get("error"):
                        errors.append(str(tr.get("error")))
        summary_lines = []
        if film_id:
            summary_lines.append(f"Film ID: {film_id}")
        if errors:
            summary_lines.append("Errors: " + "; ".join(errors)[:800])
        summary = ("\n" + "\n".join(summary_lines)) if summary_lines else ""
        status_block = (
            "\n\n### Status\n"
            "Initiated tool-based generation flow." + summary + "\n"
            "Use `film_status` to track progress and `/api/jobs` for live status."
        )
        # Always append status; never replace footer/body
        display_content = (display_content or "") + status_block
    # Build artifacts block from tool_results (prefer film.run)
    artifacts: Dict[str, Any] = {}
    try:
        film_run = None
        for tr in (tool_results or []):
            if isinstance(tr, dict) and tr.get("name") == "film.run" and isinstance(tr.get("result"), dict):
                film_run = tr.get("result")
        if film_run:
            master_uri = film_run.get("master_uri") or (film_run.get("master") or {}).get("uri")
            master_hash = film_run.get("hash") or (film_run.get("master") or {}).get("hash")
            eff = film_run.get("effective") or {}
            res_eff = eff.get("res")
            fps_eff = eff.get("refresh")
            edl = film_run.get("edl") or {}
            nodes = film_run.get("nodes") or {}
            qc = film_run.get("qc_report") or {}
            export_pkg = {"uri": film_run.get("package_uri"), "hash": film_run.get("hash")}
            artifacts = {
                "master": {"uri": master_uri, "hash": master_hash, "res": res_eff, "fps": fps_eff},
                "edl": {"uri": edl.get("uri"), "hash": edl.get("hash")},
                "nodes": {"uri": nodes.get("uri"), "hash": nodes.get("hash")},
                "qc": {"uri": qc.get("uri"), "hash": qc.get("hash")},
                "export": export_pkg,
            }
    except Exception:
        artifacts = {}

    # Build canonical envelope (merged from steps) to attach to response for internal use
    try:
        step_texts: List[str] = []
        if isinstance(plan_text, str) and plan_text.strip():
            step_texts.append(plan_text)
        if tool_results:
            try:
                step_texts.append(json.dumps(tool_results, ensure_ascii=False))
            except Exception:
                pass
        if isinstance(qwen_text, str) and qwen_text.strip():
            step_texts.append(qwen_text)
        if isinstance(gptoss_text, str) and gptoss_text.strip():
            step_texts.append(gptoss_text)
        if isinstance(display_content, str) and display_content.strip():
            step_texts.append(display_content)
        step_envs = [normalize_to_envelope(t) for t in step_texts]
        final_env = stitch_merge_envelopes(step_envs)
        # Merge tool_calls and artifacts deterministically from tool_exec_meta
        tc_merged: List[Dict[str, Any]] = []
        arts: List[Dict[str, Any]] = []
        seen_art_ids = set()
        for meta in (tool_exec_meta or []):
            name = meta.get("name")
            args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
            tc_merged.append({"tool": name, "args": args, "status": "done", "result_ref": None})
            a = meta.get("artifacts") or {}
            if isinstance(a, dict):
                for k, v in a.items():
                    if isinstance(v, dict):
                        aid = v.get("hash") or v.get("uri") or f"{name}:{k}"
                        if aid in seen_art_ids:
                            continue
                        seen_art_ids.add(aid)
                        arts.append({"id": aid, "kind": k, "summary": name or k, **{kk: vv for kk, vv in v.items() if kk in ("uri", "hash")}})
        if isinstance(final_env, dict):
            final_env["tool_calls"] = tc_merged
            final_env["artifacts"] = arts
    except Exception:
        final_env = {}

    response = {
        "id": "orc-1",
        "object": "chat.completion",
        "model": f"committee:{QWEN_MODEL_ID}+{GPTOSS_MODEL_ID}",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": display_content}
            }
        ],
        "usage": usage_with_wall,
        "seed": master_seed,
    }
    if artifacts:
        response["artifacts"] = artifacts
    if isinstance(final_env, dict) and final_env:
        # Optional ablation: extract grounded facts and export
        try:
            do_ablate = os.getenv("ABLATE", "on").lower() == "on"
            do_export = os.getenv("ABLATE_EXPORT", "on").lower() == "on"
            scope = os.getenv("ABLATE_SCOPE", "auto")
            if do_ablate:
                abl = ablate_env(final_env, scope_hint=(scope if scope != "auto" else "chat"))
                final_env["ablated"] = abl
                if do_export:
                    try:
                        outdir = os.path.join(UPLOAD_DIR, "ablation", trace_id)
                        facts_path = ablate_write_facts(abl, trace_id, outdir)
                        # attach a reference to the exported dataset
                        response.setdefault("artifacts", {})
                        if isinstance(response["artifacts"], dict):
                            uri = _uri_from_upload_path(facts_path)
                            response["artifacts"]["ablation_facts"] = {"uri": uri}
                    except Exception:
                        pass
        except Exception:
            pass
        response["envelope"] = final_env
    # Finalize artifacts shard and write a tiny manifest reference
    try:
        if _ledger_shard is not None:
            _art_finalize_shard_local = _art_finalize  # alias to avoid shadowing name
            _art_finalize_shard_local(_ledger_shard)
            _mani = {"items": []}
            idx_path = os.path.join(_ledger_root, f"{_ledger_name}.index.json")
            _art_manifest_add(_mani, idx_path, step_id="final")
            _art_manifest_write(_ledger_root, _mani)
    except Exception:
        pass

    # Fire-and-forget: Teacher trace tap (if reachable)
    try:
        # Build trace payload
        req_dict = body.dict()
        try:
            msgs_for_seed = json.dumps(req_dict.get("messages", []), ensure_ascii=False, separators=(",", ":"))
        except Exception:
            msgs_for_seed = ""
        provided_seed2 = None
        try:
            if isinstance(req_dict.get("seed"), (int, float)):
                provided_seed2 = int(req_dict.get("seed"))
        except Exception:
            provided_seed2 = None
        master_seed = provided_seed2 if provided_seed2 is not None else _derive_seed("chat", msgs_for_seed)
        label_cfg = None
        try:
            label_cfg = (WRAPPER_CONFIG.get("teacher") or {}).get("default_label")
        except Exception:
            label_cfg = None
        # Normalize tool_calls shape for teacher (use args, not arguments)
        _tc2 = tool_exec_meta or []
        trace_payload = {
            "label": label_cfg or "exp_default",
            "seed": master_seed,
            "request": {"messages": req_dict.get("messages", []), "tools_allowed": [t.get("function", {}).get("name") for t in (body.get("tools") or []) if isinstance(t, dict)]},
            "context": ({"pack_hash": pack_hash} if pack_hash else {}),
            "routing": {"planner_model": planner_id, "executors": [QWEN_MODEL_ID, GPTOSS_MODEL_ID], "seed_router": det_seed_router(trace_id, master_seed)},
            "tool_calls": _tc2,
            "response": {"text": (display_content or "")[:4000]},
            "metrics": usage,
            "env": {"public_base_url": PUBLIC_BASE_URL, "config_hash": WRAPPER_CONFIG_HASH},
            "privacy": {"vault_refs": 0, "secrets_in": 0, "secrets_out": 0},
        }
        async def _send_trace():
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    await client.post(TEACHER_API_URL.rstrip("/") + "/teacher/trace.append", json=trace_payload)
            except Exception:
                return
        asyncio.create_task(_send_trace())
    except Exception:
        pass
    # Persist response & metrics
    await _db_update_run_response(run_id, response, usage)
    try:
        _append_event(STATE_DIR, trace_id, "halt", {"kind": "committee", "chars": len(display_content)})
    except Exception:
        pass
    try:
        if _lock_token:
            _release_lock(STATE_DIR, trace_id)
    except Exception:
        pass
    return JSONResponse(content=response)


@app.post("/run")
async def run_endpoint(body: Dict[str, Any], request: Request):
    # Minimal adapter to reuse the same pipeline
    # Allow either OpenAI-compatible or plain JSON
    if isinstance(body.get("messages"), list):
        return await chat_completions(body, request)
    # Coerce a single prompt string into messages
    prompt = str(body.get("prompt") or "")
    cr = {"model": body.get("model"), "messages": [{"role": "user", "content": prompt}], "stream": bool(body.get("stream", False)), "tools": body.get("tools")}
    return await chat_completions(cr, request)


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "openai_compat": True,
        "teacher_enabled": True,
        "icw_enabled": bool((WRAPPER_CONFIG.get("icw") or {}).get("enabled", False)),
        "film_enabled": bool((WRAPPER_CONFIG.get("film") or {}).get("enabled", False)),
        "ablation_enabled": bool((WRAPPER_CONFIG.get("ablation") or {}).get("enabled", False)),
    }


# Lightweight in-memory job controls for orchestrator-run tools (research.run, etc.)
@app.get("/orcjobs/{job_id}")
async def orcjob_get(job_id: str):
    j = _get_orcjob(job_id)
    if not j:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return {
        "id": j.id,
        "tool": j.tool,
        "args": j.args,
        "state": j.state,
        "phase": j.phase,
        "progress": j.progress,
        "created_at": j.created_at,
        "updated_at": j.updated_at,
        "error": j.error,
    }


@app.post("/orcjobs/{job_id}/cancel")
async def orcjob_cancel(job_id: str):
    j = _get_orcjob(job_id)
    if not j:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    if j.state in ("done", "failed", "cancelled"):
        return {"ok": True, "state": j.state}
    _orcjob_cancel(job_id)
    return {"ok": True, "state": "cancelling"}


@app.get("/orcjobs/{job_id}/stream")
async def orcjob_stream(job_id: str, interval_ms: Optional[int] = None):
    async def _gen():
        last = None
        while True:
            j = _get_orcjob(job_id)
            if not j:
                yield "data: {\"error\": \"not_found\"}\n\n"
                break
            snapshot = json.dumps({
                "id": j.id,
                "tool": j.tool,
                "state": j.state,
                "phase": j.phase,
                "progress": j.progress,
                "updated_at": j.updated_at,
            })
            if snapshot != last:
                yield f"data: {snapshot}\n\n"
                last = snapshot
            if j.state in ("done", "failed", "cancelled"):
                yield "data: [DONE]\n\n"
                break
            await asyncio.sleep(max(0.01, (interval_ms or 1000) / 1000.0))
    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/debug")
async def debug():
    # sanity checks to both backends
    try:
        qr = await call_ollama(QWEN_BASE_URL, {"model": QWEN_MODEL_ID, "prompt": "ping", "stream": False})
        gr = await call_ollama(GPTOSS_BASE_URL, {"model": GPTOSS_MODEL_ID, "prompt": "ping", "stream": False})
        return {"qwen_ok": "response" in qr and not qr.get("error"), "gptoss_ok": "response" in gr and not gr.get("error"), "qwen_detail": qr.get("error"), "gptoss_detail": gr.get("error")}
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": f"committee:{QWEN_MODEL_ID}+{GPTOSS_MODEL_ID}", "object": "model"},
            {"id": QWEN_MODEL_ID, "object": "model"},
            {"id": GPTOSS_MODEL_ID, "object": "model"},
        ],
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    import uuid
    suffix = ""
    if "." in file.filename:
        suffix = "." + file.filename.split(".")[-1]
    name = f"{uuid.uuid4().hex}{suffix}"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(await file.read())
    url = f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{name}" if PUBLIC_BASE_URL else f"/uploads/{name}"
    return {"url": url, "name": file.filename}



# ---------- Jobs API for long ComfyUI workflows ----------
_job_endpoint: Dict[str, str] = {}
_comfy_load: Dict[str, int] = {}


def _comfy_candidates() -> List[str]:
    if COMFYUI_API_URLS:
        return COMFYUI_API_URLS
    if COMFYUI_API_URL:
        return [COMFYUI_API_URL]
    return []


async def _pick_comfy_base() -> Optional[str]:
    candidates = _comfy_candidates()
    if not candidates:
        return None
    for u in candidates:
        _comfy_load.setdefault(u, 0)
    return min(candidates, key=lambda u: _comfy_load.get(u, 0))


async def _comfy_submit_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    candidates = _comfy_candidates()
    if not candidates:
        return {"error": "COMFYUI_API_URL(S) not configured"}
    delay = COMFYUI_BACKOFF_MS
    last_err: Optional[str] = None
    for attempt in range(1, COMFYUI_MAX_RETRIES + 1):
        ordered = sorted(candidates, key=lambda u: _comfy_load.get(u, 0))
        for base in ordered:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    if _comfy_sem is not None:
                        async with _comfy_sem:
                            _comfy_load[base] = _comfy_load.get(base, 0) + 1
                            r = await client.post(base.rstrip("/") + "/prompt", json=workflow)
                    else:
                        _comfy_load[base] = _comfy_load.get(base, 0) + 1
                        r = await client.post(base.rstrip("/") + "/prompt", json=workflow)
                    try:
                        r.raise_for_status()
                        res = r.json()
                        pid = res.get("prompt_id") or res.get("uuid") or res.get("id")
                        if isinstance(pid, str):
                            _job_endpoint[pid] = base
                        return res
                    except Exception:
                        last_err = r.text
            except Exception as ex:
                last_err = str(ex)
            finally:
                _comfy_load[base] = max(0, _comfy_load.get(base, 1) - 1)
        # backoff before next round
        try:
            await asyncio.sleep(max(0.0, float(delay) / 1000.0))
        except Exception:
            pass
        delay = min(delay * 2, COMFYUI_BACKOFF_MAX_MS)
    return {"error": last_err or "all comfyui instances failed after retries"}


async def _comfy_history(prompt_id: str) -> Dict[str, Any]:
    base = _job_endpoint.get(prompt_id) or (await _pick_comfy_base())
    if not base:
        return {"error": "COMFYUI_API_URL(S) not configured"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(base.rstrip("/") + f"/history/{prompt_id}")
        try:
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"error": r.text}


async def get_film_characters(film_id: str) -> List[Dict[str, Any]]:
    pool = await get_pg_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, name, description, reference_data FROM characters WHERE film_id=$1", film_id)
    return [dict(r) for r in rows]


async def get_film_preferences(film_id: str) -> Dict[str, Any]:
    pool = await get_pg_pool()
    if pool is None:
        return {}
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT metadata FROM films WHERE id=$1", film_id)
    meta = (row and row.get("metadata")) or {}
    if isinstance(meta, dict):
        return dict(meta)
    if isinstance(meta, str):
        try:
            # Expect a preferences dict (open schema)
            parsed = JSONParser().parse(meta, {})
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_resolution(res: Optional[str]) -> Tuple[int, int]:
    try:
        if isinstance(res, str) and "x" in res:
            w, h = res.lower().split("x", 1)
            return max(64, int(w)), max(64, int(h))
    except Exception:
        pass
    return 1024, 1024


def _parse_duration_seconds_dynamic(value: Any, default_seconds: float = 10.0) -> int:
    try:
        if value is None:
            return int(default_seconds)
        if isinstance(value, (int, float)):
            return max(1, int(round(float(value))))
        s = str(value).strip().lower()
        if not s:
            return int(default_seconds)
        # HH:MM:SS or MM:SS
        import re as _re
        if _re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", s):
            parts = [int(x) for x in s.split(":")]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
        # textual: "3 minutes", "180s", "3m"
        m = _re.match(r"^(\d+)\s*(seconds|second|secs|sec|s)$", s)
        if m:
            return max(1, int(m.group(1)))
        m = _re.match(r"^(\d+)\s*(minutes|minute|mins|min|m)$", s)
        if m:
            return max(1, int(m.group(1)) * 60)
        m = _re.match(r"^(\d+)\s*(hours|hour|hrs|hr|h)$", s)
        if m:
            return max(1, int(m.group(1)) * 3600)
        # plain integer string
        return max(1, int(round(float(s))))
    except Exception:
        return int(default_seconds)


def _parse_fps_dynamic(value: Any, default_fps: int = 60) -> int:
    try:
        if value is None:
            return default_fps
        if isinstance(value, (int, float)):
            return max(1, min(240, int(round(float(value)))))
        s = str(value).strip().lower()
        import re as _re
        m = _re.match(r"^(\d{1,3})\s*fps$", s)
        if m:
            return max(1, min(240, int(m.group(1))))
        return max(1, min(240, int(round(float(s)))))
    except Exception:
        return default_fps


def build_default_scene_workflow(prompt: str, characters: List[Dict[str, Any]], style: Optional[str] = None, *, width: int = 1024, height: int = 1024, steps: int = 25, seed: int = 0, filename_prefix: str = "scene") -> Dict[str, Any]:
    # Minimal SDXL image generation graph using CheckpointLoaderSimple (MODEL, CLIP, VAE)
    positive = prompt
    # Inject lightweight story context into the prompt to reduce drift
    try:
        # Fetch film synopsis and primary character names if available
        # This function is used within film_add_scene where film_id context exists; callers pass characters list
        if characters:
            names = [c.get("name") for c in characters if isinstance(c, dict) and isinstance(c.get("name"), str)]
            if names:
                positive = f"{positive}. Characters: " + ", ".join([n for n in names[:3] if n])
    except Exception:
        pass
    if style:
        positive = f"{prompt} in {style} style"
    for ch in characters[:2]:
        name = ch.get("name")
        if name:
            positive += f", featuring {name}"
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["9", 0]}},
    }
    return {"prompt": g}


def _derive_seed(*parts: str) -> int:
    import hashlib
    h = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    # 32-bit range for Comfy samplers
    return int(h[:8], 16)


def _clamp_frames(frames: int) -> int:
    try:
        return max(1, min(int(frames), 120))
    except Exception:
        return 24


def _round_fps(fps: float) -> int:
    try:
        return max(1, min(int(round(float(fps))), 60))
    except Exception:
        return 24


def build_animated_scene_workflow(
    prompt: str,
    characters: List[Dict[str, Any]],
    *,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    style: Optional[str],
    seed: int,
    filename_prefix: str,
    upscale_enabled: bool = False,
    upscale_scale: int = 0,
    interpolation_enabled: bool = False,
    interpolation_multiplier: int = 0,
) -> Dict[str, Any]:
    # Simple animation via chunked batch sampling using CheckpointLoaderSimple
    total_frames = int(max(1, duration_seconds) * fps)
    frames = _clamp_frames(total_frames)
    batch_frames = max(1, min(SCENE_MAX_BATCH_FRAMES, frames))
    positive = prompt
    # Inject lightweight story context into the prompt to reduce drift
    try:
        names = [c.get("name") for c in characters if isinstance(c, dict) and isinstance(c.get("name"), str)]
        if names:
            positive = f"{positive}. Characters: " + ", ".join([n for n in names[:3] if n])
    except Exception:
        pass
    if style:
        positive = f"{prompt} in {style} style"
    # Build a graph that renders a smaller batch (batch_frames)
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": batch_frames}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 20, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
    }
    last_image_node = "9"
    # Optional Real-ESRGAN upscale
    if upscale_enabled and upscale_scale in (2, 3, 4):
        g["11"] = {"class_type": "RealESRGANModelLoader", "inputs": {"model_name": "realesr-general-x4v3.pth"}}
        g["12"] = {"class_type": "RealESRGAN", "inputs": {"image": [last_image_node, 0], "model": ["11", 0], "scale": upscale_scale}}
        last_image_node = "12"
    # Optional RIFE interpolation (VideoHelperSuite)
    if interpolation_enabled and interpolation_multiplier and interpolation_multiplier > 1:
        g["13"] = {"class_type": "VHS_RIFE_VFI", "inputs": {"frames": [last_image_node, 0], "multiplier": interpolation_multiplier, "model": "rife-v4.6"}}
        last_image_node = "13"
    # Final save
    g["10"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": [last_image_node, 0]}}
    return {"prompt": g, "_batch_frames": batch_frames, "_total_frames": total_frames}


async def _track_comfy_job(job_id: str, prompt_id: str) -> None:
    pool = await get_pg_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute("UPDATE jobs SET status='running', updated_at=NOW() WHERE id=$1", job_id)
    _jobs_store.setdefault(job_id, {})["state"] = "running"
    _jobs_store[job_id]["updated_at"] = time.time()
    last_outputs_count = -1
    last_error = None
    # Wait generously for history to appear; do not fail early just because history is temporarily missing
    HISTORY_GRACE_SECONDS = int(os.getenv("HISTORY_GRACE_SECONDS", "86400"))  # default: 24h
    start_time = time.time()
    while True:
        data = await _comfy_history(prompt_id)
        _jobs_store[job_id]["updated_at"] = time.time()
        _jobs_store[job_id]["last_history"] = data
        history = (data or {}).get("history", {})
        detail = history.get(prompt_id) if isinstance(history, dict) else None
        if isinstance(detail, dict):
            outputs = detail.get("outputs", {}) or {}
            outputs_count = 0
            if isinstance(outputs, dict):
                for v in outputs.values():
                    if isinstance(v, list):
                        outputs_count += len(v)
                    else:
                        outputs_count += 1
            if outputs_count != last_outputs_count and outputs_count > 0:
                last_outputs_count = outputs_count
                async with pool.acquire() as conn:
                    await conn.execute("INSERT INTO job_checkpoints (job_id, data) VALUES ($1, $2)", job_id, json.dumps(detail))
            if detail.get("status", {}).get("completed") is True:
                result = {"outputs": detail.get("outputs", {}), "status": detail.get("status")}
                async with pool.acquire() as conn:
                    await conn.execute("UPDATE jobs SET status='succeeded', updated_at=NOW(), result=$1 WHERE id=$2", json.dumps(result), job_id)
                _jobs_store[job_id]["state"] = "succeeded"
                _jobs_store[job_id]["result"] = result
                # propagate to scene if any
                try:
                    await _update_scene_from_job(job_id, detail)
                except Exception:
                    pass
                if JOBS_RAG_INDEX:
                    try:
                        await _index_job_into_rag(job_id)
                    except Exception:
                        pass
                break
            if detail.get("status", {}).get("status") == "error":
                err = detail.get("status")
                async with pool.acquire() as conn:
                    await conn.execute("UPDATE jobs SET status='failed', updated_at=NOW(), error=$1 WHERE id=$2", json.dumps(err), job_id)
                _jobs_store[job_id]["state"] = "failed"
                _jobs_store[job_id]["error"] = err
                try:
                    await _update_scene_from_job(job_id, detail, failed=True)
                except Exception:
                    pass
                break
        else:
            # History not yet available. Do NOT fail early and do NOT auto-resubmit; keep polling up to grace window.
            if isinstance(data, dict) and data.get("error"):
                last_error = data.get("error")
            if (time.time() - start_time) >= HISTORY_GRACE_SECONDS:
                # Grace window exceeded – keep job as running and attach last_error for visibility, but don't mark failed
                _jobs_store[job_id]["state"] = "running"
                if last_error:
                    _jobs_store[job_id]["error"] = last_error
                await asyncio.sleep(2.0)
                continue
            await asyncio.sleep(2.0)
            continue
        # keep polling
        await asyncio.sleep(2.0)


@app.post("/jobs")
async def create_job(body: Dict[str, Any]):
    if not COMFYUI_API_URL:
        return JSONResponse(status_code=400, content={"error": "COMFYUI_API_URL not configured"})
    workflow = body.get("workflow") if isinstance(body, dict) else None
    if not workflow:
        workflow = body or {}
    submit = await _comfy_submit_workflow(workflow)
    if submit.get("error"):
        return JSONResponse(status_code=502, content=submit)
    prompt_id = submit.get("prompt_id") or submit.get("uuid") or submit.get("id")
    if not prompt_id:
        return JSONResponse(status_code=502, content={"error": "invalid comfy response", "detail": submit})
    import uuid as _uuid
    job_id = _uuid.uuid4().hex
    pool = await get_pg_pool()
    if pool is not None:
        async with pool.acquire() as conn:
            await conn.execute("INSERT INTO jobs (id, prompt_id, status, workflow) VALUES ($1, $2, 'queued', $3)", job_id, prompt_id, json.dumps(workflow))
    _jobs_store[job_id] = {"id": job_id, "prompt_id": prompt_id, "state": "queued", "created_at": time.time(), "updated_at": time.time(), "result": None}
    asyncio.create_task(_track_comfy_job(job_id, prompt_id))
    return {"job_id": job_id, "prompt_id": prompt_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    pool = await get_pg_pool()
    if pool is None:
        job = _jobs_store.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return job
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, prompt_id, status, created_at, updated_at, workflow, result, error FROM jobs WHERE id=$1", job_id)
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        cps = await conn.fetch("SELECT id, created_at, data FROM job_checkpoints WHERE job_id=$1 ORDER BY id DESC LIMIT 10", job_id)
        return {**dict(row), "checkpoints": [dict(c) for c in cps]}


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, interval_ms: Optional[int] = None):
    async def _gen():
        last_snapshot = None
        while True:
            pool = await get_pg_pool()
            if pool is None:
                snapshot = json.dumps(_jobs_store.get(job_id) or {"error": "not found"})
            else:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("SELECT id, prompt_id, status, created_at, updated_at, workflow, result, error FROM jobs WHERE id=$1", job_id)
                    if not row:
                        yield "data: {\"error\": \"not found\"}\n\n"
                        break
                    snapshot = json.dumps(dict(row))
            if snapshot != last_snapshot:
                yield f"data: {snapshot}\n\n"
                last_snapshot = snapshot
            try:
                # Parse current job status safely via JSONParser. Never use json.loads.
                state = (JSONParser().parse(snapshot or "", {"status": str}) or {}).get("status")
            except Exception:
                state = None
            if state in ("succeeded", "failed", "cancelled"):
                yield "data: [DONE]\n\n"
                break
            await asyncio.sleep(max(0.01, (interval_ms or 1000) / 1000.0))

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    pool = await get_pg_pool()
    if pool is None:
        items = list(_jobs_store.values())
        if status:
            items = [j for j in items if j.get("state") == status]
        return {"data": items[offset: offset + limit], "total": len(items)}
    async with pool.acquire() as conn:
        if status:
            rows = await conn.fetch("SELECT id, prompt_id, status, created_at, updated_at FROM jobs WHERE status=$1 ORDER BY updated_at DESC LIMIT $2 OFFSET $3", status, limit, offset)
            total = await conn.fetchval("SELECT COUNT(*) FROM jobs WHERE status=$1", status)
        else:
            rows = await conn.fetch("SELECT id, prompt_id, status, created_at, updated_at FROM jobs ORDER BY updated_at DESC LIMIT $1 OFFSET $2", limit, offset)
            total = await conn.fetchval("SELECT COUNT(*) FROM jobs")
        db_items = [dict(r) for r in rows]
        # Union with in-memory items to avoid empty UI when DB insert fails
        db_ids = {it.get("id") for it in db_items}
        mem_items = list(_jobs_store.values())
        if status:
            mem_items = [j for j in mem_items if j.get("state") == status]
        mem_only = [j for j in mem_items if j.get("id") not in db_ids]
        all_items = db_items + mem_only
        return {"data": all_items, "total": int(total) + len([1 for j in mem_only])}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    pool = await get_pg_pool()
    if pool is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT prompt_id, status FROM jobs WHERE id=$1", job_id)
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        if row["status"] in ("succeeded", "failed", "cancelled"):
            return {"ok": True, "status": row["status"]}
        await conn.execute("UPDATE jobs SET status='cancelling', updated_at=NOW() WHERE id=$1", job_id)
    try:
        if COMFYUI_API_URL:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(COMFYUI_API_URL.rstrip("/") + "/interrupt")
    except Exception:
        pass
    async with pool.acquire() as conn:
        await conn.execute("UPDATE jobs SET status='cancelled', updated_at=NOW() WHERE id=$1", job_id)
    _jobs_store.setdefault(job_id, {})["state"] = "cancelled"
    return {"ok": True, "status": "cancelled"}


async def _index_job_into_rag(job_id: str) -> None:
    pool = await get_pg_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT workflow, result FROM jobs WHERE id=$1", job_id)
        if not row:
            return
        wf = json.dumps(row["workflow"], ensure_ascii=False) if row.get("workflow") is not None else ""
        res = json.dumps(row["result"], ensure_ascii=False) if row.get("result") is not None else ""
    embedder = get_embedder()
    pieces = [p for p in [wf, res] if p]
    async with pool.acquire() as conn:
        for chunk in pieces:
            vec = embedder.encode([chunk])[0]
            await conn.execute("INSERT INTO rag_docs (path, chunk, embedding) VALUES ($1, $2, $3)", f"job:{job_id}", chunk, list(vec))


def _extract_comfy_asset_urls(detail: Dict[str, Any] | str, base_override: Optional[str] = None) -> List[Dict[str, Any]]:
    if isinstance(detail, str):
        try:
            detail = JSONParser().parse(detail, {"outputs": dict, "status": dict})
        except Exception:
            detail = {}
    outputs = (detail or {}).get("outputs", {}) or {}
    urls: List[Dict[str, Any]] = []
    base_url = base_override or COMFYUI_API_URL
    if not base_url:
        return urls
    base = base_url.rstrip('/')
    def _url(fn: str, ftype: str, sub: Optional[str]) -> str:
        from urllib.parse import urlencode
        q = {"filename": fn, "type": ftype or "output"}
        if sub:
            q["subfolder"] = sub
        return f"{base}/view?{urlencode(q)}"
    if isinstance(outputs, dict):
        for node_id, items in outputs.items():
            # items are typically lists of {filename, type, subfolder}
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    fn = it.get("filename")
                    tp = it.get("type") or "output"
                    sub = it.get("subfolder")
                    if fn:
                        urls.append({"node": node_id, "filename": fn, "type": tp, "subfolder": sub, "url": _url(fn, tp, sub)})
    return urls


async def _update_scene_from_job(job_id: str, detail: Dict[str, Any] | str, failed: bool = False) -> None:
    pool = await get_pg_pool()
    if pool is None:
        return
    # Prefer the exact ComfyUI base that handled this job (per-job endpoint pinning)
    try:
        pid = (_jobs_store.get(job_id) or {}).get("prompt_id")
        base = _job_endpoint.get(pid) if pid else None
    except Exception:
        base = None
    if isinstance(detail, str):
        try:
            detail = JSONParser().parse(detail, {"outputs": dict, "status": dict})
        except Exception:
            detail = {}
    assets = {"outputs": (detail or {}).get("outputs", {}), "status": (detail or {}).get("status"), "urls": _extract_comfy_asset_urls(detail, base)}
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, film_id, plan FROM scenes WHERE job_id=$1", job_id)
        if not row:
            return
        if failed:
            await conn.execute("UPDATE scenes SET status='failed', updated_at=NOW(), assets=$1 WHERE id=$2", json.dumps(assets), row["id"])
        else:
            await conn.execute("UPDATE scenes SET status='succeeded', updated_at=NOW(), assets=$1 WHERE id=$2", json.dumps(assets), row["id"])
    if not failed:
        # Auto-generate TTS dialogue and background music for the scene
        scene_tts = None
        scene_music = None
        try:
            # Use prompt as provisional dialogue text; in practice, planner should generate scripts
            scene_text = (detail.get("status", {}) or {}).get("prompt", "") or ""
            if not scene_text:
                # fallback: try to use scene prompt from DB
                async with pool.acquire() as conn:
                    p = await conn.fetchval("SELECT prompt FROM scenes WHERE id=$1", row["id"])
                    scene_text = p or ""
            # derive duration preference
            duration = 12
            try:
                pl = row.get("plan") or {}
                prefs = (pl.get("preferences") if isinstance(pl, dict) else None) or {}
                if prefs.get("duration_seconds"):
                    duration = int(float(prefs.get("duration_seconds")))
            except Exception:
                pass
            # read toggles
            audio_enabled = True
            subs_enabled = False
            try:
                if isinstance(pl, dict):
                    prefs = (pl.get("preferences") if isinstance(pl.get("preferences"), dict) else {})
                    audio_enabled = bool(prefs.get("audio_enabled", True))
                    subs_enabled = bool(prefs.get("subtitles_enabled", False))
            except Exception:
                pass
            # choose voice: character-specific -> scene plan -> film-level preference
            voice = None
            try:
                if isinstance(pl, dict):
                    prefs = (pl.get("preferences") if isinstance(pl.get("preferences"), dict) else {})
                    voice = prefs.get("voice")
                    # character-specific override if a single character is targeted
                    ch_ids = pl.get("character_ids") if isinstance(pl.get("character_ids"), list) else []
                    if len(ch_ids) == 1:
                        async with pool.acquire() as conn:
                            crow = await conn.fetchrow("SELECT reference_data FROM characters WHERE id=$1", ch_ids[0])
                            if crow:
                                rd = crow.get("reference_data")
                                if isinstance(rd, str):
                                    try:
                                        rd = JSONParser().parse(rd, {})
                                    except Exception:
                                        rd = {}
                                if isinstance(rd, dict):
                                    v2 = rd.get("voice")
                                    if v2:
                                        voice = v2
                if not voice:
                    async with pool.acquire() as conn:
                        meta = await conn.fetchval("SELECT metadata FROM films WHERE id=$1", row["film_id"]) 
                        if isinstance(meta, dict):
                            voice = meta.get("voice")
            except Exception:
                pass
            # language preference
            language = None
            try:
                async with pool.acquire() as conn:
                    meta = await conn.fetchval("SELECT metadata FROM films WHERE id=$1", row["film_id"]) 
                    if isinstance(meta, dict):
                        language = meta.get("language")
            except Exception:
                pass
            if XTTS_API_URL and ALLOW_TOOL_EXECUTION and scene_text and audio_enabled:
                async with httpx.AsyncClient(timeout=180) as client:
                    tr = await client.post(XTTS_API_URL.rstrip("/") + "/tts", json={"text": scene_text, "voice": voice, "language": language})
                    if tr.status_code == 200:
                        scene_tts = tr.json()
            if MUSIC_API_URL and ALLOW_TOOL_EXECUTION and audio_enabled:
                async with httpx.AsyncClient(timeout=600) as client:
                    mr = await client.post(MUSIC_API_URL.rstrip("/") + "/generate", json={"prompt": "cinematic background score", "duration": duration})
                    if mr.status_code == 200:
                        scene_music = mr.json()
            # simple SRT creation from scene_text if enabled
            if scene_text and subs_enabled:
                srt = _text_to_simple_srt(scene_text, duration)
                assets["subtitles_srt"] = srt
        except Exception:
            pass
        # Update scene assets with audio
        if scene_tts or scene_music:
            assets = dict(assets)
            if scene_tts:
                assets["tts"] = scene_tts
            if scene_music:
                assets["music"] = scene_music
            async with pool.acquire() as conn:
                await conn.execute("UPDATE scenes SET assets=$1 WHERE id=$2", json.dumps(assets), row["id"])
        try:
            await _maybe_compile_film(row["film_id"])
        except Exception:
            pass


def _mean_normalize_embedding(embs: List[List[float]]) -> List[float]:
    if not embs:
        return []
    dim = len(embs[0])
    sums = [0.0] * dim
    for v in embs:
        for i in range(min(dim, len(v))):
            sums[i] += float(v[i])
    mean = [x / len(embs) for x in sums]
    # L2 normalize
    import math
    norm = math.sqrt(sum(x * x for x in mean)) or 1.0
    return [x / norm for x in mean]


def _text_to_simple_srt(text: str, duration_seconds: int) -> str:
    # naive: split by sentences, distribute time evenly
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]
    n = max(1, len(sentences))
    dur = max(1.0, float(duration_seconds))
    per = dur / n
    def fmt(t: float) -> str:
        h = int(t // 3600); t -= 3600*h
        m = int(t // 60); t -= 60*m
        s = int(t); ms = int((t - s) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, s in enumerate(sentences, 1):
        start = per * (i - 1)
        end = per * i
        lines.append(str(i))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(s)
        lines.append("")
    return "\n".join(lines)


async def _maybe_compile_film(film_id: str) -> None:
    pool = await get_pg_pool()
    if pool is None:
        return
    # If all scenes for film are succeeded, trigger compilation via N8N if available
    async with pool.acquire() as conn:
        counts = await conn.fetchrow("SELECT COUNT(*) AS total, SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS done FROM scenes WHERE film_id=$1", film_id)
        total = int(counts["total"] or 0)
        done = int(counts["done"] or 0)
        if total == 0 or done != total:
            return
        rows = await conn.fetch("SELECT id, index_num, assets FROM scenes WHERE film_id=$1 ORDER BY index_num ASC", film_id)
    scenes = [dict(r) for r in rows]
    payload = {"film_id": film_id, "scenes": [dict(s) for s in scenes], "public_base_url": PUBLIC_BASE_URL}
    assembly_result = None
    # Prefer local assembler if available
    if ASSEMBLER_API_URL:
        try:
            async with httpx.AsyncClient(timeout=1200) as client:
                r = await client.post(ASSEMBLER_API_URL.rstrip("/") + "/assemble", json=payload)
                r.raise_for_status()
                assembly_result = r.json()
        except Exception:
            assembly_result = {"error": True}
    elif N8N_WEBHOOK_URL and ENABLE_N8N:
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                r = await client.post(N8N_WEBHOOK_URL, json=payload)
                r.raise_for_status()
                assembly_result = r.json()
        except Exception:
            assembly_result = {"error": True}
    # Always write a manifest to uploads for convenience
    manifest_url = None
    try:
        manifest = json.dumps(payload)
        if EXECUTOR_BASE_URL:
            async with httpx.AsyncClient(timeout=15) as client:
                res = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/write_file", json={"path": "uploads/film_" + film_id + "_manifest.json", "content": manifest})
                res.raise_for_status()
                # Build public URL
                name = f"film_{film_id}_manifest.json"
                manifest_url = (f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{name}" if PUBLIC_BASE_URL else f"/uploads/{name}")
    except Exception:
        pass
    # persist compile artifact into films.metadata
    async with pool.acquire() as conn:
        meta_row = await conn.fetchrow("SELECT metadata FROM films WHERE id=$1", film_id)
        metadata = (meta_row and meta_row.get("metadata")) or {}
        metadata = dict(metadata)
        metadata["compiled_at"] = time.time()
        if assembly_result is not None:
            metadata["assembly"] = assembly_result
        if manifest_url:
            metadata["manifest_url"] = manifest_url
        metadata["scenes_count"] = total
        await conn.execute("UPDATE films SET metadata=$1, updated_at=NOW() WHERE id=$2", json.dumps(metadata), film_id)


# ---------- Film project endpoints ----------

