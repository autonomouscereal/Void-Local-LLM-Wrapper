from __future__ import annotations

import os
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
import time

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11434")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen2.5:32b-instruct-q4_K_M")
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
ALLOW_TOOL_EXECUTION = os.getenv("ALLOW_TOOL_EXECUTION", "true").lower() == "true"
AUTO_EXECUTE_TOOLS = os.getenv("AUTO_EXECUTE_TOOLS", "true").lower() == "true"
STREAM_CHUNK_SIZE_CHARS = int(os.getenv("STREAM_CHUNK_SIZE_CHARS", "0"))
STREAM_CHUNK_INTERVAL_MS = int(os.getenv("STREAM_CHUNK_INTERVAL_MS", "50"))
JOBS_RAG_INDEX = os.getenv("JOBS_RAG_INDEX", "true").lower() == "true"

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
XTTS_API_URL = os.getenv("XTTS_API_URL")      # e.g., http://xtts:8020
WHISPER_API_URL = os.getenv("WHISPER_API_URL")# e.g., http://whisper:9090
FACEID_API_URL = os.getenv("FACEID_API_URL")  # e.g., http://faceid:7000
MUSIC_API_URL = os.getenv("MUSIC_API_URL")    # e.g., http://musicgen:7860
VLM_API_URL = os.getenv("VLM_API_URL")        # e.g., http://vlm:8050
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")  # optional external workflow orchestration
ASSEMBLER_API_URL = os.getenv("ASSEMBLER_API_URL")  # http://assembler:9095


class ChatMessage(BaseModel):
    role: str
    content: Any | None = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    user: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    finish_reason: Optional[str] = None
    message: Dict[str, Any]


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[ChatChoice]
    model: str
    usage: Optional[Dict[str, int]] = None


app = FastAPI(title="Void Orchestrator", version="0.1.0")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# In-memory job cache (DB is source of truth)
_jobs_store: Dict[str, Dict[str, Any]] = {}


def build_ollama_payload(messages: List[ChatMessage], model: str, num_ctx: int, temperature: float) -> Dict[str, Any]:
    rendered: List[str] = []
    for m in messages:
        if m.role == "tool":
            tool_name = m.name or "tool"
            rendered.append(f"tool[{tool_name}]: {m.content}")
        else:
            if isinstance(m.content, list):
                text_parts: List[str] = []
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                content = "\n".join(text_parts)
            else:
                content = m.content if m.content is not None else ""
            rendered.append(f"{m.role}: {content}")
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


def estimate_usage(messages: List[ChatMessage], completion_text: str) -> Dict[str, int]:
    prompt_text = "\n".join([(m.content or "") for m in messages])
    prompt_tokens = estimate_tokens_from_text(prompt_text)
    completion_tokens = estimate_tokens_from_text(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


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


def extract_attachments_from_messages(messages: List[ChatMessage]) -> Tuple[List[ChatMessage], List[Dict[str, Any]]]:
    attachments: List[Dict[str, Any]] = []
    normalized: List[ChatMessage] = []
    for m in messages:
        if isinstance(m.content, list):
            text_parts: List[str] = []
            for part in m.content:
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
            normalized.append(ChatMessage(role=m.role, content=merged_text or None, name=m.name, tool_call_id=m.tool_call_id, tool_calls=m.tool_calls))
        else:
            normalized.append(m)
    return normalized, attachments


# ---------- RAG (pgvector) ----------
_engine: Optional[Engine] = None
_embedder: Optional[SentenceTransformer] = None
_rag_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def get_engine() -> Optional[Engine]:
    global _engine
    if _engine is not None:
        return _engine
    if not (POSTGRES_HOST and POSTGRES_DB and POSTGRES_USER and POSTGRES_PASSWORD):
        return None
    url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    _engine = create_engine(url, pool_pre_ping=True)
    with _engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS rag_docs (
              id BIGSERIAL PRIMARY KEY,
              path TEXT,
              chunk TEXT,
              embedding vector(384)
            );
            """
        ))
        conn.execute(text("CREATE INDEX IF NOT EXISTS rag_docs_embedding_idx ON rag_docs USING ivfflat (embedding vector_cosine) WITH (lists = 100);"))
        conn.execute(text(
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
        ))
        conn.execute(text("CREATE INDEX IF NOT EXISTS jobs_status_updated_idx ON jobs (status, updated_at DESC);"))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS job_checkpoints (
              id BIGSERIAL PRIMARY KEY,
              job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              data JSONB
            );
            """
        ))
        conn.execute(text(
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
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS characters (
              id TEXT PRIMARY KEY,
              film_id TEXT REFERENCES films(id) ON DELETE CASCADE,
              name TEXT,
              description TEXT,
              references JSONB,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ))
        conn.execute(text(
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
        ))
    return _engine


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


async def rag_index_dir(root: str = "/workspace", glob_exts: Optional[List[str]] = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    import glob as _glob
    import os as _os
    engine = get_engine()
    if engine is None:
        return {"error": "pgvector not configured"}
    exts = glob_exts or ["*.md", "*.py", "*.ts", "*.tsx", "*.js", "*.json", "*.txt"]
    files: List[str] = []
    for ext in exts:
        files.extend(_glob.glob(_os.path.join(root, "**", ext), recursive=True))
    embedder = get_embedder()
    total_chunks = 0
    with engine.begin() as conn:
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
                conn.execute(text("INSERT INTO rag_docs (path, chunk, embedding) VALUES (:p, :c, :e)"), {"p": fp.replace(root + "/", ""), "c": chunk, "e": list(vec)})
                total_chunks += 1
    return {"indexed_files": len(files), "chunks": total_chunks}


async def rag_search(query: str, k: int = 8) -> List[Dict[str, Any]]:
    now = time.time()
    key = f"{query}::{k}"
    cached = _rag_cache.get(key)
    if cached and (now - cached[0] <= RAG_CACHE_TTL_SEC):
        return cached[1]
    engine = get_engine()
    if engine is None:
        return []
    vec = get_embedder().encode([query])[0]
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT path, chunk FROM rag_docs ORDER BY embedding <=> :q LIMIT :k"), {"q": list(vec), "k": k}).fetchall()
        results = [{"path": r[0], "chunk": r[1]} for r in rows]
        _rag_cache[key] = (now, results)
        return results


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=2))
async def call_ollama(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            resp = await client.post(f"{base_url}/api/generate", json=payload)
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


async def propose_search_queries(messages: List[ChatMessage]) -> List[str]:
    guidance = (
        "Given the conversation, propose up to 3 concise web search queries that would most improve the answer. "
        "Return each query on its own line with no extra text."
    )
    prompt_messages = messages + [ChatMessage(role="user", content=guidance)]
    payload = build_ollama_payload(prompt_messages, QWEN_MODEL_ID, DEFAULT_NUM_CTX, DEFAULT_TEMPERATURE)
    try:
        result = await call_ollama(QWEN_BASE_URL, payload)
        text = result.get("response", "")
        lines = [ln.strip("- ") for ln in text.splitlines() if ln.strip()]
        # take up to 3 non-empty queries
        return lines[:3]
    except Exception:
        return []


def meta_prompt(messages: List[ChatMessage]) -> List[ChatMessage]:
    system_preface = (
        "You are part of a two-model team with explicit roles: a Planner and Executors. "
        "Planner decomposes the task, chooses tools, and requests relevant evidence. Executors produce solutions and critiques. "
        "Be precise, show working when non-trivial, and output correct, runnable code when appropriate."
    )
    out = [ChatMessage(role="system", content=system_preface)]
    # If the request suggests filmmaking, nudge to use the film tools
    try:
        text = "\n".join([
            (m.content if isinstance(m.content, str) else "\n".join([str(p.get("text","")) for p in (m.content or []) if isinstance(p, dict) and p.get("type")=="text"]))
            for m in messages
        ]).lower()
    except Exception:
        text = ""
    keywords = ("film", "movie", "scene", "character", "screenplay", "storyboard", "animation")
    if any(k in text for k in keywords):
        out.append(ChatMessage(role="system", content=(
            "For filmmaking tasks, prefer these tools: make_movie (one-shot), film_create, film_add_character, film_add_scene, film_status, film_compile. "
            "If given a single prompt, use make_movie to orchestrate end-to-end generation. "
            "Defaults: audio ON, subtitles OFF; only ask if unclear or conflicting."
        )))
        # Heuristic: if user hints at high fidelity or high fps, proactively set preferences
        hiq_terms = ("4k", "8k", "uhd", "ultra hd", "hdr", "high quality", "super high quality", "60fps", "120fps", "buttery", "smooth", "cinematic 60", "crystal clear")
        if any(t in text for t in hiq_terms):
            out.append(ChatMessage(role="system", content=(
                "If the user indicates high fidelity or high fps (e.g., 4K/8K/60fps/cinematic/smooth), set preferences accordingly: "
                "resolution=3840x2160 (or higher if explicitly requested), fps=60 if requested, quality=high, "
                "interpolation_enabled=true with interpolation_target_fps matching requested fps, "
                "upscale_enabled=true with upscale_scale=2 or 4 as needed. "
                "Keep toggles OFF by default unless intent is clear; ask at most one clarifying question only if blocking."
            )))
    return out + messages


def merge_responses(request: ChatRequest, qwen_text: str, gptoss_text: str) -> str:
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
                "name": "film_create",
                "description": "Create a new film project with an optional title and synopsis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "synopsis": {"type": "string"},
                        "metadata": {"type": "object", "description": "Optional preferences: duration_seconds, resolution, fps, style, language, voice, audio_enabled (bool), subtitles_enabled (bool), animation_enabled (bool)"}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "film_add_character",
                "description": "Add a character to a film, optionally with references (images/embeddings).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "film_id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "references": {"type": "object", "description": "e.g., {image_url: string, embedding: number[]}"}
                    },
                    "required": ["film_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "film_add_scene",
                "description": "Create a scene for the film and launch generation job via ComfyUI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "film_id": {"type": "string"},
                        "index_num": {"type": "integer"},
                        "prompt": {"type": "string"},
                        "character_ids": {"type": "array", "items": {"type": "string"}},
                        "plan": {"type": "object"},
                        "workflow": {"type": "object"},
                        "duration_seconds": {"type": "number"},
                        "style": {"type": "string"},
                        "audio_enabled": {"type": "boolean"},
                        "subtitles_enabled": {"type": "boolean"}
                    },
                    "required": ["film_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "film_compile",
                "description": "Collect scene assets and optionally hand off to N8N webhook for assembly.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "film_id": {"type": "string"}
                    },
                    "required": ["film_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "film_status",
                "description": "Return film, characters, scenes and job statuses.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "film_id": {"type": "string"}
                    },
                    "required": ["film_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "make_movie",
                "description": "High-level: from a single prompt, create a film with characters and scenes and launch generation jobs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "synopsis": {"type": "string"},
                        "characters": {"type": "array", "items": {"type": "object"}},
                        "scenes": {"type": "array", "items": {"type": "object"}},
                        "duration_seconds": {"type": "number"},
                        "resolution": {"type": "string"},
                        "fps": {"type": "number"},
                        "style": {"type": "string"},
                        "language": {"type": "string"},
                        "voice": {"type": "string"},
                        "audio_enabled": {"type": "boolean"},
                        "subtitles_enabled": {"type": "boolean"}
                    },
                    "required": []
                }
            }
        }
    ]


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

async def planner_produce_plan(messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]], temperature: float) -> Tuple[str, List[Dict[str, Any]]]:
    planner_id = QWEN_MODEL_ID if PLANNER_MODEL.lower() == "qwen" else GPTOSS_MODEL_ID
    planner_base = QWEN_BASE_URL if PLANNER_MODEL.lower() == "qwen" else GPTOSS_BASE_URL
    guide = (
        "You are the Planner. Produce a short step-by-step plan and, if helpful, propose up to 3 tool invocations.\n"
        "Ask 1-3 clarifying questions ONLY if blocking details are missing (e.g., duration, style, language, target resolution).\n"
        "If not blocked, proceed and choose reasonable defaults: duration<=10s for short clips, 1920x1080, 24fps, language=en, a neutral voice.\n"
        "Return strict JSON with keys: plan (string), tool_calls (array of {name: string, arguments: object})."
    )
    all_tools = merge_tool_schemas(tools)
    tool_info = build_tools_section(all_tools)
    plan_messages = messages + [ChatMessage(role="user", content=guide + ("\n" + tool_info if tool_info else ""))]
    payload = build_ollama_payload(plan_messages, planner_id, DEFAULT_NUM_CTX, temperature)
    result = await call_ollama(planner_base, payload)
    text = result.get("response", "").strip()
    json_text = text
    if "```" in text:
        # try to extract code block
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            candidate = parts[i]
            if candidate.lstrip().startswith("{"):
                json_text = candidate
                break
    plan = ""
    tool_calls: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(json_text)
        plan = parsed.get("plan", "")
        tool_calls = parsed.get("tool_calls", []) or []
    except Exception:
        plan = text
        tool_calls = []
    return plan, tool_calls


async def execute_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name")
    args = call.get("arguments") or {}
    if name == "web_search" and ENABLE_WEBSEARCH and SERPAPI_API_KEY and ALLOW_TOOL_EXECUTION:
        query = args.get("query") or ""
        if not query:
            return {"name": name, "error": "missing query"}
        snippets = await serpapi_google_search([query], max_results=int(args.get("k", 5)))
        return {"name": name, "result": snippets}
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
    if name == "film_create":
        engine = get_engine()
        if engine is None:
            return {"name": name, "error": "pg not configured"}
        import uuid as _uuid
        film_id = _uuid.uuid4().hex
        title = args.get("title") or "Untitled"
        synopsis = args.get("synopsis") or ""
        metadata = args.get("metadata") or {}
        # normalize preferences with sensible defaults
        prefs = {
            "duration_seconds": float(metadata.get("duration_seconds") or args.get("duration_seconds") or 10),
            "resolution": str(metadata.get("resolution") or args.get("resolution") or "1920x1080"),
            "fps": float(metadata.get("fps") or args.get("fps") or 24),
            "style": metadata.get("style") or args.get("style"),
            "language": metadata.get("language") or args.get("language") or "en",
            "voice": metadata.get("voice") or args.get("voice") or None,
            "quality": metadata.get("quality") or "standard",
        "audio_enabled": bool(metadata.get("audio_enabled") if metadata.get("audio_enabled") is not None else (args.get("audio_enabled") if args.get("audio_enabled") is not None else True)),
        "subtitles_enabled": bool(metadata.get("subtitles_enabled") if metadata.get("subtitles_enabled") is not None else (args.get("subtitles_enabled") if args.get("subtitles_enabled") is not None else False)),
        "animation_enabled": bool(metadata.get("animation_enabled") if metadata.get("animation_enabled") is not None else (args.get("animation_enabled") if args.get("animation_enabled") is not None else True)),
        }
        # attach preferences back to metadata
        metadata = {**metadata, **{k: v for k, v in prefs.items() if v is not None}}
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO films (id, title, synopsis, metadata) VALUES (:id, :t, :s, :m)"), {"id": film_id, "t": title, "s": synopsis, "m": metadata})
        return {"name": name, "result": {"film_id": film_id, "title": title}}
    if name == "film_add_character":
        engine = get_engine()
        if engine is None:
            return {"name": name, "error": "pg not configured"}
        import uuid as _uuid
        char_id = _uuid.uuid4().hex
        film_id = args.get("film_id")
        if not film_id:
            return {"name": name, "error": "missing film_id"}
        # optional face embeddings if multiple references provided; fuse intelligently
        refs = args.get("references") or {}
        urls: List[str] = []
        if isinstance(refs, dict):
            # accept single and multiple keys
            if refs.get("image_url"):
                urls.append(refs.get("image_url"))
            if isinstance(refs.get("image_urls"), list):
                urls.extend([u for u in refs.get("image_urls") if isinstance(u, str)])
            if isinstance(refs.get("attachment_urls"), list):
                urls.extend([u for u in refs.get("attachment_urls") if isinstance(u, str)])
            if isinstance(refs.get("images"), list):
                for it in refs.get("images"):
                    if isinstance(it, dict):
                        u = it.get("image_url") or it.get("url")
                        if isinstance(u, str):
                            urls.append(u)
        # dedupe
        urls = list(dict.fromkeys([u for u in urls if u]))
        face_embeddings: List[List[float]] = []
        if urls and FACEID_API_URL and ALLOW_TOOL_EXECUTION:
            for u in urls[:12]:
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        rr = await client.post(FACEID_API_URL.rstrip("/") + "/embed", json={"image_url": u})
                        rr.raise_for_status()
                        emb = (rr.json() or {}).get("embeddings") or []
                        if emb and isinstance(emb[0], list):
                            face_embeddings.append(emb[0])
                except Exception:
                    continue
        if face_embeddings:
            fused = _mean_normalize_embedding(face_embeddings)
            if isinstance(refs, dict):
                refs = {**refs, "face_embeddings": face_embeddings, "face_embedding_mean": fused}
        # else keep refs as-is
        with engine.begin() as conn:
            exists = conn.execute(text("SELECT 1 FROM films WHERE id=:id"), {"id": film_id}).first()
            if not exists:
                return {"name": name, "error": "film not found"}
            conn.execute(text("INSERT INTO characters (id, film_id, name, description, references) VALUES (:id, :f, :n, :d, :r)"), {"id": char_id, "f": film_id, "n": args.get("name") or "Unnamed", "d": args.get("description") or "", "r": refs})
        return {"name": name, "result": {"character_id": char_id}}
    if name == "film_add_scene":
        engine = get_engine()
        if engine is None:
            return {"name": name, "error": "pg not configured"}
        film_id = args.get("film_id")
        if not film_id:
            return {"name": name, "error": "missing film_id"}
        prompt = args.get("prompt") or ""
        index_num = int(args.get("index_num", 0))
        character_ids = args.get("character_ids") or []
        plan = args.get("plan") or {}
        workflow = args.get("workflow") or plan
        advanced_used = False
        if not workflow:
            prefs = _get_film_preferences(engine, film_id)
            res_w, res_h = _parse_resolution(prefs.get("resolution") or "1024x1024")
            steps = 25
            if isinstance(prefs.get("quality"), str) and prefs.get("quality") == "high":
                steps = 35
            chars = _get_film_characters(engine, film_id)
            # derive stable seed from film+scene meta for appearance stability
            seed = _derive_seed(film_id, str(index_num), prompt)
            filename_prefix = f"scene_{index_num:03d}"
            if bool(prefs.get("animation_enabled", True)):
                fps = _round_fps(prefs.get("fps") or 24)
                duration = float(prefs.get("duration_seconds") or 4)
                # Determine upscale/interpolation prefs
                up_enabled = bool(prefs.get("upscale_enabled", False))
                up_scale = int(prefs.get("upscale_scale") or 0)
                it_enabled = bool(prefs.get("interpolation_enabled", False))
                # Convert target fps to multiplier if desired
                it_multiplier = 0
                try:
                    target_fps = int(prefs.get("interpolation_target_fps") or 0)
                    if target_fps and target_fps > fps:
                        it_multiplier = max(2, min(4, int(round(target_fps / fps))))
                except Exception:
                    it_multiplier = 0
                workflow = build_animated_scene_workflow(
                    prompt=prompt,
                    characters=chars,
                    width=res_w,
                    height=res_h,
                    fps=fps,
                    duration_seconds=duration,
                    style=(args.get("style") or prefs.get("style") or None),
                    seed=seed,
                    filename_prefix=filename_prefix,
                    upscale_enabled=up_enabled,
                    upscale_scale=up_scale,
                    interpolation_enabled=it_enabled,
                    interpolation_multiplier=it_multiplier,
                )
                advanced_used = True
            else:
                workflow = build_default_scene_workflow(
                    prompt=prompt,
                    characters=chars,
                    style=(args.get("style") or prefs.get("style") or None),
                    width=res_w,
                    height=res_h,
                    steps=steps,
                    seed=seed,
                    filename_prefix=filename_prefix,
                )
        submit = await _comfy_submit_workflow(workflow)
        if submit.get("error") and advanced_used:
            # graceful fallback to basic still workflow if advanced nodes unavailable
            prefs = _get_film_preferences(engine, film_id)
            res_w, res_h = _parse_resolution(prefs.get("resolution") or "1024x1024")
            steps = 25
            if isinstance(prefs.get("quality"), str) and prefs.get("quality") == "high":
                steps = 35
            seed = _derive_seed(film_id, str(index_num), prompt)
            filename_prefix = f"scene_{index_num:03d}"
            workflow = build_default_scene_workflow(
                prompt=prompt,
                characters=_get_film_characters(engine, film_id),
                style=(args.get("style") or prefs.get("style") or None),
                width=res_w,
                height=res_h,
                steps=steps,
                seed=seed,
                filename_prefix=filename_prefix,
            )
            submit = await _comfy_submit_workflow(workflow)
        if submit.get("error"):
            return {"name": name, "error": submit.get("error")}
        prompt_id = submit.get("prompt_id") or submit.get("uuid") or submit.get("id")
        import uuid as _uuid
        scene_id = _uuid.uuid4().hex
        job_id = _uuid.uuid4().hex
        # resolve toggles for audio/subtitles
        pref_audio = args.get("audio_enabled")
        pref_subs = args.get("subtitles_enabled")
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO jobs (id, prompt_id, status, workflow) VALUES (:id, :pid, 'queued', :wf)"), {"id": job_id, "pid": prompt_id, "wf": workflow})
            # merge scene-local toggles into preferences
            merged_prefs = dict(prefs) if isinstance(prefs, dict) else {}
            if pref_audio is not None:
                merged_prefs["audio_enabled"] = bool(pref_audio)
            if pref_subs is not None:
                merged_prefs["subtitles_enabled"] = bool(pref_subs)
            # include character_ids into plan for downstream consistency
            scene_plan = {**plan, "preferences": merged_prefs, "character_ids": character_ids}
            conn.execute(text("INSERT INTO scenes (id, film_id, index_num, prompt, plan, status, job_id) VALUES (:id, :f, :idx, :p, :pl, 'queued', :jid)"), {"id": scene_id, "f": film_id, "idx": index_num, "p": prompt, "pl": scene_plan, "jid": job_id})
        _jobs_store[job_id] = {"id": job_id, "prompt_id": prompt_id, "state": "queued", "created_at": time.time(), "updated_at": time.time()}
        asyncio.create_task(_track_comfy_job(job_id, prompt_id))
        return {"name": name, "result": {"scene_id": scene_id, "job_id": job_id}}
    if name == "film_compile":
        engine = get_engine()
        if engine is None:
            return {"name": name, "error": "pg not configured"}
        film_id = args.get("film_id")
        if not film_id:
            return {"name": name, "error": "missing film_id"}
        with engine.begin() as conn:
            scenes = conn.execute(text("SELECT id, index_num, assets FROM scenes WHERE film_id=:f ORDER BY index_num ASC"), {"f": film_id}).mappings().all()
        payload = {"film_id": film_id, "scenes": [dict(s) for s in scenes], "public_base_url": PUBLIC_BASE_URL}
        if N8N_WEBHOOK_URL:
            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    r = await client.post(N8N_WEBHOOK_URL, json=payload)
                    r.raise_for_status()
                    return {"name": name, "result": r.json()}
            except Exception as ex:
                return {"name": name, "error": str(ex)}
        return {"name": name, "result": payload}
    if name == "film_status":
        engine = get_engine()
        if engine is None:
            return {"name": name, "error": "pg not configured"}
        film_id = args.get("film_id")
        if not film_id:
            return {"name": name, "error": "missing film_id"}
        with engine.begin() as conn:
            film = conn.execute(text("SELECT id, title, synopsis, created_at, updated_at FROM films WHERE id=:id"), {"id": film_id}).mappings().first()
            if not film:
                return {"name": name, "error": "not found"}
            chars = conn.execute(text("SELECT id, name, description FROM characters WHERE film_id=:f"), {"f": film_id}).mappings().all()
            scenes = conn.execute(text("SELECT id, index_num, status, job_id FROM scenes WHERE film_id=:f ORDER BY index_num"), {"f": film_id}).mappings().all()
        return {"name": name, "result": {"film": dict(film), "characters": [dict(c) for c in chars], "scenes": [dict(s) for s in scenes]}}
    if name == "make_movie":
        # High-level convenience: create film, add characters if provided, create scenes from plan
        title = args.get("title") or "Untitled"
        synopsis = args.get("synopsis") or args.get("prompt") or ""
        characters = args.get("characters") or []
        scenes = args.get("scenes") or []
        movie_prefs = {k: args.get(k) for k in ("duration_seconds","resolution","fps","style","language","voice") if args.get(k) is not None}
        # create film
        fc = await execute_tool_call({"name": "film_create", "arguments": {"title": title, "synopsis": synopsis, "metadata": movie_prefs}})
        if fc.get("error"):
            return {"name": name, "error": fc.get("error")}
        film_id = (fc.get("result") or {}).get("film_id")
        # add characters
        for ch in characters[:25]:
            await execute_tool_call({"name": "film_add_character", "arguments": {"film_id": film_id, **({} if not isinstance(ch, dict) else ch)}})
        # add scenes
        created = []
        for sc in scenes[:200]:
            if not isinstance(sc, dict):
                sc = {"prompt": str(sc)}
            res = await execute_tool_call({"name": "film_add_scene", "arguments": {"film_id": film_id, **sc}})
            created.append(res)
        return {"name": name, "result": {"film_id": film_id, "created": created}}
    if name == "run_python" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/run_python", json={"code": args.get("code", "")})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "write_file" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/write_file", json={"path": args.get("path"), "content": args.get("content", "")})
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
            results.append({"name": call.get("name", "unknown"), "error": str(ex)})
    return results


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest, request: Request):
    # normalize and extract attachments (images/audio/video/files) for tools
    normalized_msgs, attachments = extract_attachments_from_messages(body.messages)
    # prepend a system hint with attachment summary (non-invasive)
    if attachments:
        attn = json.dumps(attachments, indent=2)
        normalized_msgs = [ChatMessage(role="system", content=f"Attachments available for tools:\n{attn}")] + normalized_msgs
    messages = meta_prompt(normalized_msgs)

    # If client supplies tool results (role=tool) we include them verbatim for the planner/executors

    # Optional self-ask-with-search augmentation
    if ENABLE_WEBSEARCH and SERPAPI_API_KEY:
        queries = await propose_search_queries(messages)
        if queries:
            snippets = await serpapi_google_search(queries, max_results=5)
            if snippets:
                messages = [
                    ChatMessage(role="system", content="Web search results follow. Use them only if relevant."),
                    ChatMessage(role="system", content=snippets),
                ] + messages

    # 1) Planner proposes plan + tool calls
    plan_text, tool_calls = await planner_produce_plan(messages, body.tools, body.temperature or DEFAULT_TEMPERATURE)
    # tool_choice=required compatibility: force at least one tool_call if tools are provided
    if (body.tool_choice == "required") and (not tool_calls) and body.tools:
        # choose first declared tool with empty args
        first = body.tools[0]
        fn = (first.get("function") or {}).get("name") or first.get("name") or "tool"
        tool_calls = [{"name": fn, "arguments": {}}]

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
        if body.stream:
            async def _stream_once():
                chunk = json.dumps({"id": response["id"], "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": tool_calls_openai}, "finish_reason": None}]})
                yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream_once(), media_type="text/event-stream")
        # include usage estimate even in tool_calls path (no completion tokens yet)
        response["usage"] = estimate_usage(messages, "")
        return JSONResponse(content=response)

    # 2) Optionally execute tools
    tool_results: List[Dict[str, Any]] = []
    if tool_calls:
        tool_results = await execute_tools(tool_calls)

    # 3) Executors respond independently using plan + evidence
    evidence_blocks: List[ChatMessage] = []
    if plan_text:
        evidence_blocks.append(ChatMessage(role="system", content=f"Planner plan:\n{plan_text}"))
    if tool_results:
        evidence_blocks.append(ChatMessage(role="system", content="Tool results:\n" + json.dumps(tool_results, indent=2)))
    exec_messages = evidence_blocks + messages

    qwen_payload = build_ollama_payload(
        messages=exec_messages, model=QWEN_MODEL_ID, num_ctx=DEFAULT_NUM_CTX, temperature=body.temperature or DEFAULT_TEMPERATURE
    )
    gptoss_payload = build_ollama_payload(
        messages=exec_messages, model=GPTOSS_MODEL_ID, num_ctx=DEFAULT_NUM_CTX, temperature=body.temperature or DEFAULT_TEMPERATURE
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
        return JSONResponse(status_code=502, content={"error": "backend_failed", "detail": detail})

    qwen_text = qwen_result.get("response", "")
    gptoss_text = gptoss_result.get("response", "")

    # 4) Optional brief debate (cross-critique)
    if ENABLE_DEBATE and MAX_DEBATE_TURNS > 0:
        critique_prompt = (
            "Critique the other model's answer. Identify mistakes, missing considerations, or improvements. "
            "Return a concise bullet list."
        )
        qwen_critique_msg = exec_messages + [ChatMessage(role="user", content=critique_prompt + f"\nOther answer:\n{gptoss_text}")]
        gptoss_critique_msg = exec_messages + [ChatMessage(role="user", content=critique_prompt + f"\nOther answer:\n{qwen_text}")]
        qwen_crit_payload = build_ollama_payload(qwen_critique_msg, QWEN_MODEL_ID, DEFAULT_NUM_CTX, body.temperature or DEFAULT_TEMPERATURE)
        gptoss_crit_payload = build_ollama_payload(gptoss_critique_msg, GPTOSS_MODEL_ID, DEFAULT_NUM_CTX, body.temperature or DEFAULT_TEMPERATURE)
        qcrit_task = asyncio.create_task(call_ollama(QWEN_BASE_URL, qwen_crit_payload))
        gcrit_task = asyncio.create_task(call_ollama(GPTOSS_BASE_URL, gptoss_crit_payload))
        qcrit_res, gcrit_res = await asyncio.gather(qcrit_task, gcrit_task)
        qcrit_text = qcrit_res.get("response", "")
        gcrit_text = gcrit_res.get("response", "")
        exec_messages = exec_messages + [
            ChatMessage(role="system", content="Cross-critique from Qwen:\n" + qcrit_text),
            ChatMessage(role="system", content="Cross-critique from GPT-OSS:\n" + gcrit_text),
        ]

    # 5) Final synthesis by Planner
    final_request = exec_messages + [
        ChatMessage(
            role="user",
            content=(
                "Produce the final, corrected answer, incorporating critiques and evidence. "
                "Be unambiguous, include runnable code when requested, and prefer specific citations to tool results."
            ),
        )
    ]

    planner_id = QWEN_MODEL_ID if PLANNER_MODEL.lower() == "qwen" else GPTOSS_MODEL_ID
    planner_base = QWEN_BASE_URL if PLANNER_MODEL.lower() == "qwen" else GPTOSS_BASE_URL
    synth_payload = build_ollama_payload(final_request, planner_id, DEFAULT_NUM_CTX, body.temperature or DEFAULT_TEMPERATURE)
    synth_result = await call_ollama(planner_base, synth_payload)
    final_text = synth_result.get("response", "") or qwen_text or gptoss_text

    if body.stream:
        async def _stream_final(text: str):
            head = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            yield f"data: {head}\n\n"
            if STREAM_CHUNK_SIZE_CHARS and STREAM_CHUNK_SIZE_CHARS > 0:
                size = max(1, STREAM_CHUNK_SIZE_CHARS)
                for i in range(0, len(text), size):
                    piece = text[i : i + size]
                    chunk = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                    yield f"data: {chunk}\n\n"
                    if STREAM_CHUNK_INTERVAL_MS > 0:
                        await asyncio.sleep(STREAM_CHUNK_INTERVAL_MS / 1000.0)
            else:
                content_chunk = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
                yield f"data: {content_chunk}\n\n"
            done = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield f"data: {done}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream_final(final_text), media_type="text/event-stream")

    # Merge exact usages if available, else approximate
    usage = merge_usages([
        qwen_result.get("_usage"),
        gptoss_result.get("_usage"),
        synth_result.get("_usage"),
    ])
    if usage["total_tokens"] == 0:
        usage = estimate_usage(messages, final_text)

    response = ChatResponse(
        id="orc-1",
        model=f"committee:{QWEN_MODEL_ID}+{GPTOSS_MODEL_ID}",
        choices=[
            ChatChoice(index=0, finish_reason="stop", message={"role": "assistant", "content": final_text})
        ],
        usage=usage,
    )
    return JSONResponse(content=response.model_dump())


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


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
async def _comfy_submit_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    if not COMFYUI_API_URL:
        return {"error": "COMFYUI_API_URL not configured"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json=workflow)
        try:
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"error": r.text}


async def _comfy_history(prompt_id: str) -> Dict[str, Any]:
    if not COMFYUI_API_URL:
        return {"error": "COMFYUI_API_URL not configured"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(COMFYUI_API_URL.rstrip("/") + f"/history/{prompt_id}")
        try:
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"error": r.text}


def _get_film_characters(engine: Engine, film_id: str) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, name, description, references FROM characters WHERE film_id=:f"), {"f": film_id}).mappings().all()
    return [dict(r) for r in rows]


def _get_film_preferences(engine: Engine, film_id: str) -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT metadata FROM films WHERE id=:id"), {"id": film_id}).mappings().first()
    meta = (row and row.get("metadata")) or {}
    return dict(meta) if isinstance(meta, dict) else {}


def _parse_resolution(res: Optional[str]) -> Tuple[int, int]:
    try:
        if isinstance(res, str) and "x" in res:
            w, h = res.lower().split("x", 1)
            return max(64, int(w)), max(64, int(h))
    except Exception:
        pass
    return 1024, 1024


def build_default_scene_workflow(prompt: str, characters: List[Dict[str, Any]], style: Optional[str] = None, *, width: int = 1024, height: int = 1024, steps: int = 25, seed: int = 0, filename_prefix: str = "scene") -> Dict[str, Any]:
    # Minimal SDXL image generation graph with optional IP-Adapter face conditioning when available
    # This is a safe fallback workflow; more advanced animated workflows can replace this.
    positive = prompt
    if style:
        positive = f"{prompt} in {style} style"
    # Optionally incorporate character hints
    for ch in characters[:2]:
        name = ch.get("name")
        if name:
            positive += f", featuring {name}"
    # ComfyUI graph
    g = {
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["5", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["5", 0]}},
        "5": {"class_type": "LoadCLIP", "inputs": {"clip_name": "clip_g"}},
        "6": {"class_type": "Load Checkpoint", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["6", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAE Decode", "inputs": {"samples": ["8", 0], "vae": ["6", 2]}},
        "10": {"class_type": "Save Image", "inputs": {"filename_prefix": filename_prefix, "images": ["9", 0]}},
    }
    # If a character has a face_embedding, hint via positive text; proper IP-Adapter nodes can be added in a richer graph
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
    # Simple animation via batch sampling: generate N frames in a batch and save sequentially.
    # This is a safe, minimal fallback until advanced AnimateDiff graphs are provided.
    frames = _clamp_frames(int(max(1, duration_seconds) * fps))
    positive = prompt
    if style:
        positive = f"{prompt} in {style} style"
    g = {
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["5", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["5", 0]}},
        "5": {"class_type": "LoadCLIP", "inputs": {"clip_name": "clip_g"}},
        "6": {"class_type": "Load Checkpoint", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": frames}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 20, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["6", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAE Decode", "inputs": {"samples": ["8", 0], "vae": ["6", 2]}},
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
    g["10"] = {"class_type": "Save Image", "inputs": {"filename_prefix": filename_prefix, "images": [last_image_node, 0]}}
    return {"prompt": g}


async def _track_comfy_job(job_id: str, prompt_id: str) -> None:
    engine = get_engine()
    if engine is None:
        return
    with engine.begin() as conn:
        conn.execute(text("UPDATE jobs SET status='running', updated_at=NOW() WHERE id=:id"), {"id": job_id})
    _jobs_store.setdefault(job_id, {})["state"] = "running"
    _jobs_store[job_id]["updated_at"] = time.time()
    last_outputs_count = -1
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
                with engine.begin() as conn:
                    conn.execute(text("INSERT INTO job_checkpoints (job_id, data) VALUES (:jid, :data)"), {"jid": job_id, "data": detail})
            if detail.get("status", {}).get("completed") is True:
                result = {"outputs": detail.get("outputs", {}), "status": detail.get("status")}
                with engine.begin() as conn:
                    conn.execute(text("UPDATE jobs SET status='succeeded', updated_at=NOW(), result=:res WHERE id=:id"), {"id": job_id, "res": result})
                _jobs_store[job_id]["state"] = "succeeded"
                _jobs_store[job_id]["result"] = result
                # propagate to scene if any
                try:
                    await _update_scene_from_job(engine, job_id, detail)
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
                with engine.begin() as conn:
                    conn.execute(text("UPDATE jobs SET status='failed', updated_at=NOW(), error=:err WHERE id=:id"), {"id": job_id, "err": json.dumps(err)})
                _jobs_store[job_id]["state"] = "failed"
                _jobs_store[job_id]["error"] = err
                try:
                    await _update_scene_from_job(engine, job_id, detail, failed=True)
                except Exception:
                    pass
                break
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
    engine = get_engine()
    if engine is not None:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO jobs (id, prompt_id, status, workflow) VALUES (:id, :pid, 'queued', :wf)"), {"id": job_id, "pid": prompt_id, "wf": workflow})
    _jobs_store[job_id] = {"id": job_id, "prompt_id": prompt_id, "state": "queued", "created_at": time.time(), "updated_at": time.time(), "result": None}
    asyncio.create_task(_track_comfy_job(job_id, prompt_id))
    return {"job_id": job_id, "prompt_id": prompt_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    engine = get_engine()
    if engine is None:
        job = _jobs_store.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return job
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id, prompt_id, status, created_at, updated_at, workflow, result, error FROM jobs WHERE id=:id"), {"id": job_id}).mappings().first()
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        cps = conn.execute(text("SELECT id, created_at, data FROM job_checkpoints WHERE job_id=:id ORDER BY id DESC LIMIT 10"), {"id": job_id}).mappings().all()
        return {**dict(row), "checkpoints": [dict(c) for c in cps]}


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, interval_ms: Optional[int] = None):
    async def _gen():
        last_snapshot = None
        while True:
            engine = get_engine()
            if engine is None:
                snapshot = json.dumps(_jobs_store.get(job_id) or {"error": "not found"})
            else:
                with engine.begin() as conn:
                    row = conn.execute(text("SELECT id, prompt_id, status, created_at, updated_at, workflow, result, error FROM jobs WHERE id=:id"), {"id": job_id}).mappings().first()
                    if not row:
                        yield "data: {\"error\": \"not found\"}\n\n"
                        break
                    snapshot = json.dumps(dict(row))
            if snapshot != last_snapshot:
                yield f"data: {snapshot}\n\n"
                last_snapshot = snapshot
            state = (json.loads(snapshot) or {}).get("status")
            if state in ("succeeded", "failed", "cancelled"):
                yield "data: [DONE]\n\n"
                break
            await asyncio.sleep(max(0.01, (interval_ms or 1000) / 1000.0))

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    engine = get_engine()
    if engine is None:
        items = list(_jobs_store.values())
        if status:
            items = [j for j in items if j.get("state") == status]
        return {"data": items[offset: offset + limit], "total": len(items)}
    with engine.begin() as conn:
        if status:
            rows = conn.execute(text("SELECT id, prompt_id, status, created_at, updated_at FROM jobs WHERE status=:s ORDER BY updated_at DESC LIMIT :l OFFSET :o"), {"s": status, "l": limit, "o": offset}).mappings().all()
            total = conn.execute(text("SELECT COUNT(*) FROM jobs WHERE status=:s"), {"s": status}).scalar_one()
        else:
            rows = conn.execute(text("SELECT id, prompt_id, status, created_at, updated_at FROM jobs ORDER BY updated_at DESC LIMIT :l OFFSET :o"), {"l": limit, "o": offset}).mappings().all()
            total = conn.execute(text("SELECT COUNT(*) FROM jobs")).scalar_one()
        return {"data": [dict(r) for r in rows], "total": int(total)}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    with engine.begin() as conn:
        row = conn.execute(text("SELECT prompt_id, status FROM jobs WHERE id=:id"), {"id": job_id}).mappings().first()
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        if row["status"] in ("succeeded", "failed", "cancelled"):
            return {"ok": True, "status": row["status"]}
        conn.execute(text("UPDATE jobs SET status='cancelling', updated_at=NOW() WHERE id=:id"), {"id": job_id})
    try:
        if COMFYUI_API_URL:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(COMFYUI_API_URL.rstrip("/") + "/interrupt")
    except Exception:
        pass
    with engine.begin() as conn:
        conn.execute(text("UPDATE jobs SET status='cancelled', updated_at=NOW() WHERE id=:id"), {"id": job_id})
    _jobs_store.setdefault(job_id, {})["state"] = "cancelled"
    return {"ok": True, "status": "cancelled"}


async def _index_job_into_rag(job_id: str) -> None:
    engine = get_engine()
    if engine is None:
        return
    with engine.begin() as conn:
        row = conn.execute(text("SELECT workflow, result FROM jobs WHERE id=:id"), {"id": job_id}).mappings().first()
        if not row:
            return
        wf = json.dumps(row["workflow"], ensure_ascii=False)
        res = json.dumps(row["result"], ensure_ascii=False)
    embedder = get_embedder()
    pieces = [p for p in [wf, res] if p]
    with engine.begin() as conn:
        for chunk in pieces:
            vec = embedder.encode([chunk])[0]
            conn.execute(text("INSERT INTO rag_docs (path, chunk, embedding) VALUES (:p, :c, :e)"), {"p": f"job:{job_id}", "c": chunk, "e": list(vec)})


def _extract_comfy_asset_urls(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs = (detail or {}).get("outputs", {}) or {}
    urls: List[Dict[str, Any]] = []
    if not COMFYUI_API_URL:
        return urls
    base = COMFYUI_API_URL.rstrip('/')
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


async def _update_scene_from_job(engine: Engine, job_id: str, detail: Dict[str, Any], failed: bool = False) -> None:
    assets = {"outputs": (detail or {}).get("outputs", {}), "status": (detail or {}).get("status"), "urls": _extract_comfy_asset_urls(detail)}
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id, film_id, plan FROM scenes WHERE job_id=:jid"), {"jid": job_id}).mappings().first()
        if not row:
            return
        if failed:
            conn.execute(text("UPDATE scenes SET status='failed', updated_at=NOW(), assets=:a WHERE id=:id"), {"id": row["id"], "a": assets})
        else:
            conn.execute(text("UPDATE scenes SET status='succeeded', updated_at=NOW(), assets=:a WHERE id=:id"), {"id": row["id"], "a": assets})
    if not failed:
        # Auto-generate TTS dialogue and background music for the scene
        scene_tts = None
        scene_music = None
        try:
            # Use prompt as provisional dialogue text; in practice, planner should generate scripts
            scene_text = (detail.get("status", {}) or {}).get("prompt", "") or ""
            if not scene_text:
                # fallback: try to use scene prompt from DB
                with engine.begin() as conn:
                    p = conn.execute(text("SELECT prompt FROM scenes WHERE id=:id"), {"id": row["id"]}).scalar_one_or_none()
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
                        with engine.begin() as conn:
                            crow = conn.execute(text("SELECT references FROM characters WHERE id=:id"), {"id": ch_ids[0]}).mappings().first()
                            if crow and isinstance(crow.get("references"), dict):
                                v2 = crow["references"].get("voice")
                                if v2:
                                    voice = v2
                if not voice:
                    with engine.begin() as conn:
                        meta = conn.execute(text("SELECT metadata FROM films WHERE id=:id"), {"id": row["film_id"]}).scalar_one_or_none()
                        if isinstance(meta, dict):
                            voice = meta.get("voice")
            except Exception:
                pass
            # language preference
            language = None
            try:
                with engine.begin() as conn:
                    meta = conn.execute(text("SELECT metadata FROM films WHERE id=:id"), {"id": row["film_id"]}).scalar_one_or_none()
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
            with engine.begin() as conn:
                conn.execute(text("UPDATE scenes SET assets=:a WHERE id=:id"), {"id": row["id"], "a": assets})
        try:
            await _maybe_compile_film(engine, row["film_id"])
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


async def _maybe_compile_film(engine: Engine, film_id: str) -> None:
    # If all scenes for film are succeeded, trigger compilation via N8N if available
    with engine.begin() as conn:
        counts = conn.execute(text("SELECT COUNT(*) AS total, SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS done FROM scenes WHERE film_id=:f"), {"f": film_id}).mappings().first()
        total = int(counts["total"] or 0)
        done = int(counts["done"] or 0)
        if total == 0 or done != total:
            return
        scenes = conn.execute(text("SELECT id, index_num, assets FROM scenes WHERE film_id=:f ORDER BY index_num ASC"), {"f": film_id}).mappings().all()
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
    elif N8N_WEBHOOK_URL:
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
    with engine.begin() as conn:
        # read current metadata
        meta_row = conn.execute(text("SELECT metadata FROM films WHERE id=:id"), {"id": film_id}).mappings().first()
        metadata = (meta_row and meta_row.get("metadata")) or {}
        metadata = dict(metadata)
        metadata["compiled_at"] = time.time()
        if assembly_result is not None:
            metadata["assembly"] = assembly_result
        if manifest_url:
            metadata["manifest_url"] = manifest_url
        metadata["scenes_count"] = total
        conn.execute(text("UPDATE films SET metadata=:m, updated_at=NOW() WHERE id=:id"), {"id": film_id, "m": metadata})


# ---------- Film project endpoints ----------
@app.post("/films")
async def create_film(body: Dict[str, Any]):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    import uuid as _uuid
    film_id = _uuid.uuid4().hex
    title = (body or {}).get("title") or "Untitled"
    synopsis = (body or {}).get("synopsis") or ""
    metadata = (body or {}).get("metadata") or {}
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO films (id, title, synopsis, metadata) VALUES (:id, :t, :s, :m)"), {"id": film_id, "t": title, "s": synopsis, "m": metadata})
    return {"id": film_id, "title": title}


@app.get("/films")
async def list_films(limit: int = 50, offset: int = 0):
    engine = get_engine()
    if engine is None:
        return {"data": [], "total": 0}
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, title, synopsis, created_at, updated_at FROM films ORDER BY updated_at DESC LIMIT :l OFFSET :o"), {"l": limit, "o": offset}).mappings().all()
        total = conn.execute(text("SELECT COUNT(*) FROM films")).scalar_one()
        return {"data": [dict(r) for r in rows], "total": int(total)}


@app.get("/films/{film_id}")
async def get_film(film_id: str):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    with engine.begin() as conn:
        film = conn.execute(text("SELECT id, title, synopsis, metadata, created_at, updated_at FROM films WHERE id=:id"), {"id": film_id}).mappings().first()
        if not film:
            return JSONResponse(status_code=404, content={"error": "not found"})
    return dict(film)


@app.patch("/films/{film_id}")
async def update_film_preferences(film_id: str, body: Dict[str, Any]):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    # merge metadata
    updates = (body or {}).get("metadata") or body or {}
    with engine.begin() as conn:
        row = conn.execute(text("SELECT metadata FROM films WHERE id=:id"), {"id": film_id}).mappings().first()
    if not row:
        return JSONResponse(status_code=404, content={"error": "not found"})
    current = row.get("metadata") or {}
    if not isinstance(current, dict):
        current = {}
    merged = {**current, **updates}
    with engine.begin() as conn:
        conn.execute(text("UPDATE films SET metadata=:m, updated_at=NOW() WHERE id=:id"), {"id": film_id, "m": merged})
    return {"ok": True, "metadata": merged}


@app.get("/films/{film_id}/characters")
async def get_characters(film_id: str):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, name, description, references FROM characters WHERE film_id=:f"), {"f": film_id}).mappings().all()
    return {"data": [dict(r) for r in rows]}


@app.get("/films/{film_id}/scenes")
async def get_scenes(film_id: str):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, index_num, prompt, status, job_id, assets FROM scenes WHERE film_id=:f ORDER BY index_num"), {"f": film_id}).mappings().all()
    return {"data": [dict(r) for r in rows]}


@app.patch("/scenes/{scene_id}")
async def update_scene(scene_id: str, body: Dict[str, Any]):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    updates = body or {}
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id, plan, prompt, index_num FROM scenes WHERE id=:id"), {"id": scene_id}).mappings().first()
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        plan = row["plan"] or {}
        if updates.get("plan"):
            plan = {**plan, **updates.get("plan")}
        if updates.get("prompt") is not None:
            conn.execute(text("UPDATE scenes SET prompt=:p, plan=:pl, updated_at=NOW() WHERE id=:id"), {"id": scene_id, "p": updates.get("prompt"), "pl": plan})
        else:
            conn.execute(text("UPDATE scenes SET plan=:pl, updated_at=NOW() WHERE id=:id"), {"id": scene_id, "pl": plan})
    return {"ok": True}


@app.post("/films/{film_id}/characters")
async def add_character(film_id: str, body: Dict[str, Any]):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    import uuid as _uuid
    char_id = _uuid.uuid4().hex
    name = (body or {}).get("name") or "Unnamed"
    description = (body or {}).get("description") or ""
    references = (body or {}).get("references") or {}
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO characters (id, film_id, name, description, references) VALUES (:id, :f, :n, :d, :r)"), {"id": char_id, "f": film_id, "n": name, "d": description, "r": references})
    return {"id": char_id, "name": name}


@app.patch("/characters/{character_id}")
async def update_character(character_id: str, body: Dict[str, Any]):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    updates = body or {}
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id, film_id, name, description, references FROM characters WHERE id=:id"), {"id": character_id}).mappings().first()
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        new_name = updates.get("name", row["name"])
        new_desc = updates.get("description", row["description"])
        refs = updates.get("references")
        if refs is None:
            refs = row["references"]
        conn.execute(text("UPDATE characters SET name=:n, description=:d, references=:r, updated_at=NOW() WHERE id=:id"), {"id": character_id, "n": new_name, "d": new_desc, "r": refs})
    return {"ok": True}


@app.post("/films/{film_id}/scenes")
async def create_scene(film_id: str, body: Dict[str, Any]):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    prompt = (body or {}).get("prompt") or ""
    index_num = int((body or {}).get("index_num") or 0)
    plan = (body or {}).get("plan") or {}
    # fire a ComfyUI job using the plan as workflow
    workflow = (body or {}).get("workflow") or plan
    submit = await _comfy_submit_workflow(workflow)
    if submit.get("error"):
        return JSONResponse(status_code=502, content=submit)
    prompt_id = submit.get("prompt_id") or submit.get("uuid") or submit.get("id")
    import uuid as _uuid
    scene_id = _uuid.uuid4().hex
    job_id = _uuid.uuid4().hex
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO jobs (id, prompt_id, status, workflow) VALUES (:id, :pid, 'queued', :wf)"), {"id": job_id, "pid": prompt_id, "wf": workflow})
        conn.execute(text("INSERT INTO scenes (id, film_id, index_num, prompt, plan, status, job_id) VALUES (:id, :f, :idx, :p, :pl, 'queued', :jid)"), {"id": scene_id, "f": film_id, "idx": index_num, "p": prompt, "pl": plan, "jid": job_id})
    _jobs_store[job_id] = {"id": job_id, "prompt_id": prompt_id, "state": "queued", "created_at": time.time(), "updated_at": time.time()}
    asyncio.create_task(_track_comfy_job(job_id, prompt_id))
    return {"id": scene_id, "job_id": job_id}


@app.post("/films/{film_id}/compile")
async def compile_film(film_id: str):
    engine = get_engine()
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    # Gather scenes and assets
    with engine.begin() as conn:
        scenes = conn.execute(text("SELECT id, index_num, assets FROM scenes WHERE film_id=:f ORDER BY index_num ASC"), {"f": film_id}).mappings().all()
        meta = conn.execute(text("SELECT metadata FROM films WHERE id=:id"), {"id": film_id}).scalar_one_or_none()
    payload = {"film_id": film_id, "scenes": [dict(s) for s in scenes], "public_base_url": PUBLIC_BASE_URL, "preferences": (meta or {})}
    # Optionally hand off to n8n for compilation pipeline
    if N8N_WEBHOOK_URL:
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                r = await client.post(N8N_WEBHOOK_URL, json=payload)
                r.raise_for_status()
                return {"ok": True, "n8n": r.json()}
        except Exception as ex:
            return JSONResponse(status_code=502, content={"error": str(ex)})
    # If no n8n, return payload for client-side assembly
    return {"ok": True, "payload": payload}

