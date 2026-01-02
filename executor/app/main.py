from __future__ import annotations
# HARD BAN (permanent): No Pydantic, no SQLAlchemy/ORM, no CSV/Parquet. JSON/NDJSON only.

import os
import json
import subprocess
import tempfile
import textwrap
import shutil
from typing import Any, Dict, Optional
import time
import urllib.request
import urllib.error
import traceback
import logging, sys
from logging.handlers import RotatingFileHandler
sys.path.append("/workspace")
from void_json import JSONParser  # shared hardened JSON parser
from void_artifacts import build_artifact, Artifact  # shared artifact builder and dataclass

import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from executor.utc.utc_runner import utc_run_tool  # moved to top per import policy
from executor.utc.db import ensure_tables
import uuid


WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
EXEC_MEMORY_MB = int(os.getenv("EXEC_MEMORY_MB", "2048"))
ALLOW_SHELL = os.getenv("ALLOW_SHELL", "false").lower() == "true"
SHELL_WHITELIST = set([s for s in (os.getenv("SHELL_WHITELIST") or "").split(",") if s])
MAX_TOOL_ATTEMPTS = int(os.getenv("EXECUTOR_MAX_ATTEMPTS", "3"))


app = FastAPI(title="Void Executor", version="0.1.0")
try:
    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "executor.log")
    _lvl = getattr(logging, (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_lvl,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(_log_file, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger(__name__).info("executor logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger(__name__).warning("executor file logging disabled: %s", _ex, exc_info=True)
log = logging.getLogger(__name__)




@app.on_event("startup")
async def _startup():
    # Let startup failures surface normally; no swallowing.
    await ensure_tables()
    logging.info("[executor] UTC tables ensured")

def within_workspace(path: str):
    full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    # Guard against path traversal; callers should surface this as an error envelope.
    if not full.startswith(os.path.abspath(WORKSPACE_DIR)):
        return os.path.abspath(WORKSPACE_DIR)
    return full


def run_subprocess(cmd: list[str], cwd: Optional[str] = None):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or WORKSPACE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return {"returncode": proc.returncode, "stdout": out, "stderr": err}


def _banned_reason_from_code(code: str):
    c = (code or "").lower()
    banned_libs = (
        ("pydantic", ("import pydantic", "from pydantic")),
        ("sqlalchemy", ("import sqlalchemy", "from sqlalchemy")),
        ("pandas", ("import pandas", "from pandas", "to_csv(", "read_csv(", "to_parquet(", "read_parquet(")),
        ("pyarrow", ("import pyarrow", "from pyarrow")),
        ("fastparquet", ("import fastparquet", "from fastparquet")),
        ("polars", ("import polars", "from polars")),
        ("django.orm", ("from django.db",)),
        ("peewee", ("import peewee", "from peewee")),
        ("tortoise-orm", ("from tortoise", "import tortoise")),
    )
    for name, needles in banned_libs:
        if any(n in c for n in needles):
            return name
    return None


def _banned_reason_from_shell(cmd: str):
    c = (cmd or "").lower()
    if "pip install" in c or "pip3 install" in c:
        banned = ("pydantic", "sqlalchemy", "pandas", "pyarrow", "fastparquet", "polars", "peewee", "tortoise")
        hits = [b for b in banned if b in c]
        if hits:
            return ",".join(hits)
    return None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/run_python")
async def run_python(body: Dict[str, Any]):
    code = body.get("code") or ""
    if not code:
        return JSONResponse(status_code=400, content={"error": "missing code"})
    reason = _banned_reason_from_code(code)
    if reason:
        return JSONResponse(status_code=422, content={"error": "forbidden_library", "detail": f"use of banned library detected: {reason}"})
    with tempfile.TemporaryDirectory(dir=WORKSPACE_DIR) as tmpd:
        path = os.path.join(tmpd, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        res = run_subprocess(["python", path], cwd=tmpd)
        return res


@app.post("/run_shell")
async def run_shell(body: Dict[str, Any]):
    if not ALLOW_SHELL:
        return JSONResponse(status_code=403, content={"error": "shell disabled"})
    cmd = body.get("cmd")
    if not cmd:
        return JSONResponse(status_code=400, content={"error": "missing cmd"})
    reason = _banned_reason_from_shell(cmd)
    if reason:
        return JSONResponse(status_code=422, content={"error": "forbidden_library", "detail": f"pip install of banned library detected: {reason}"})
    if SHELL_WHITELIST and (cmd.split(" ")[0] not in SHELL_WHITELIST):
        return JSONResponse(status_code=403, content={"error": "command not whitelisted"})
    res = run_subprocess(["bash", "-lc", cmd])
    return res


@app.post("/write_file")
async def write_file(body: Dict[str, Any]):
    rel = body.get("path")
    content = body.get("content", "")
    if not rel:
        return JSONResponse(status_code=400, content={"error": "missing path"})
    full = within_workspace(rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True}


@app.post("/read_file")
async def read_file(body: Dict[str, Any]):
    rel = body.get("path")
    if not rel:
        return JSONResponse(status_code=400, content={"error": "missing path"})
    full = within_workspace(rel)
    if not os.path.exists(full):
        return JSONResponse(status_code=404, content={"error": "not found"})
    with open(full, "r", encoding="utf-8") as f:
        return {"content": f.read()}


ORCHESTRATOR_BASE_URL = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")


def _post_json(url: str, payload: Dict[str, Any], expected: Optional[Dict[str, Any]] = None):
    """
    Post JSON to orchestrator.

    Logging is critical: if orchestrator ingestion fails, we still emit the
    payload to executor logs so the event isn't lost, and we do NOT crash /execute.
    """
    import urllib.error

    data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            body = (e.read() or b"").decode("utf-8", errors="replace")
        except Exception:
            body = ""
        logging.error(
            f"[executor] _post_json HTTPError url={url!r} status={getattr(e, 'code', None)!r} "
            f"body_prefix={(body or '')[:300]!r} payload_keys={sorted(list(payload.keys()))[:64]!r}"
        )
        # Preserve the event locally (stdout/file) so it is not lost.
        logging.error("[executor] _post_json payload=%s", json.dumps(payload, ensure_ascii=False, default=str))
        return {}
    except urllib.error.URLError as e:
        logging.error("[executor] _post_json URLError url=%r err=%r payload_keys=%r", url, e, sorted(list(payload.keys()))[:64])
        logging.error("[executor] _post_json payload=%r", json.dumps(payload, ensure_ascii=False, default=str))
        return {}

    parser = JSONParser()
    schema = expected if expected is not None else {}
    try:
        parsed = parser.parse(raw, schema)
        if not isinstance(parsed, dict):
            logging.warning("_post_json: JSONParser returned non-dict type=%s url=%s", type(parsed).__name__, url)
            return {}
        return parsed
    except Exception as ex:
        logging.warning("_post_json: JSONParser.parse failed url=%s ex=%s raw_prefix=%s", url, ex, (raw or "")[:200], exc_info=True)
        return {}


def _distill_summary(result_obj: Any):
    if not isinstance(result_obj, dict):
        return {"type": type(result_obj).__name__}
    out: Dict[str, Any] = {"keys": sorted([k for k in result_obj.keys()][:16])}
    if isinstance(result_obj.get("images"), list):
        out["images_count"] = len(result_obj.get("images") or [])
    if isinstance(result_obj.get("paths"), list):
        out["paths_count"] = len(result_obj.get("paths") or [])
    if isinstance(result_obj.get("prompt_id"), str):
        out["prompt_id"] = result_obj.get("prompt_id")
    if isinstance(result_obj.get("wav_bytes"), (bytes, bytearray)):
        out["wav_bytes"] = len(result_obj.get("wav_bytes") or b"")
    # Heuristics for video/audio artifacts in tool results
    if isinstance(result_obj.get("videos"), list):
        out["videos_count"] = len(result_obj.get("videos") or [])
    # Inspect generic lists that may contain file-like entries
    for key in ("files", "outputs"):
        val = result_obj.get(key)
        if isinstance(val, list):
            safe_vals = [x for x in val if isinstance(x, (str, bytes, bytearray))]
            exts = [
                (x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)).lower()
                for x in safe_vals
            ]
            vids = [x for x in exts if any(x.endswith(e) for e in (".mp4", ".mov", ".mkv", ".webm"))]
            auds = [x for x in exts if any(x.endswith(e) for e in (".wav", ".mp3", ".flac", ".aac"))]
            if vids:
                out["videos_count"] = max(int(out.get("videos_count") or 0), len(vids))
            if auds:
                out["audio_files_count"] = max(int(out.get("audio_files_count") or 0), len(auds))
    return out


def _build_executor_envelope(
    *,
    trace_id: str,
    conversation_id: str,
    ok: bool,
    step_outputs_by_step_id: Dict[str, Any] | None,
    error: Dict[str, Any] | None,
):
    """
    Canonical executor envelope builder.

    This mirrors the orchestrator ToolEnvelope shape (trace_id required):
    {
      "schema_version": 1,
      "trace_id": "...",
      "ok": bool,
      "result": {"produced": {...}},
      "error": {... or None}
    }
    The caller is responsible for ensuring error.message/traceback contain the
    full stack trace when ok is False.
    """
    # Derive the final ok flag from both the explicit ok hint and any per-step
    # tool results. If any step reports a non-2xx status, the overall executor
    # run is considered failed, and callers must inspect .error/.result for
    # details instead of relying on transport status.
    final_ok = bool(ok)
    try:
        for step_result in (step_outputs_by_step_id or {}).values():
            if not isinstance(step_result, dict):
                continue
            res = step_result.get("result")
            if not isinstance(res, dict):
                continue
            st = res.get("status")
            if isinstance(st, int) and st != 200:
                final_ok = False
                break
    except Exception:
        # Never let envelope construction fail; fall back to the provided ok hint.
        final_ok = bool(ok)
    trace_id = str(trace_id or "").strip() if isinstance(trace_id, str) else ""
    conversation_id = str(conversation_id or "").strip() if isinstance(conversation_id, str) else ""
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "ok": final_ok,
        "result": {"produced": step_outputs_by_step_id or {}},
        "error": error,
    }


def _canonical_tool_result(x: Dict[str, Any] | Any):
    """
    Normalize diverse tool result shapes into a minimal but faithful structure.

    Historically we only preserved ``ids`` and ``meta`` which caused richer
    fields like ``artifacts`` (image/video paths, etc.) and ``qa`` to be
    silently dropped for tools such as ``image.dispatch``. This helper now
    passes those through so downstream callers (planner, UI) can rely on them.
    """
    data: Dict[str, Any] = {}
    if isinstance(x, dict) and "ok" in x and isinstance(x.get("result"), dict):
        # ToolEnvelope-style: {ok, result: {...}, error: ...}
        data = x.get("result") or {}
    elif isinstance(x, dict):
        # Bare result object from a tool
        data = x

    external_ids = data.get("ids") if isinstance(data.get("ids"), dict) else {}  # external API identifiers
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}

    # Lift common meta-like fields from top-level bare results to prevent accidental drops
    lift_keys = (
        "url",
        "view_url",
        "poster_data_url",
        "preview_url",
        "mime",
        "duration_s",
        "preview_duration_s",
        "data_url",
    )
    for k in lift_keys:
        if k in data and k not in meta:
            v = data.get(k)
            if isinstance(v, (str, int, float, bool)) or v is None:
                meta[k] = v

    # Preserve simple identifiers that callers commonly rely on.
    # Tools may surface conversation_id at the top level; keep it in meta to avoid drops.
    meta["conversation_id"] = data.get("conversation_id")
    if "project_id" in data and "project_id" not in external_ids and isinstance(data.get("project_id"), str):
        external_ids["project_id"] = data.get("project_id")

    # Heuristic normalization for single-artifact tools that don't emit an explicit
    # artifacts list (e.g. services/music, some audio/video helpers).
    artifacts = data.get("artifacts")
    if isinstance(artifacts, list):
        # Normalize existing artifacts using Artifact.from_dict to ensure proper structure
        normalized_artifacts = []
        trace_id_meta = data.get("trace_id") or meta.get("trace_id") or ""
        conversation_id_meta = data.get("conversation_id") or meta.get("conversation_id")
        tool_name_meta = data.get("tool_name") or data.get("tool")
        for a in artifacts:
            if isinstance(a, dict):
                try:
                    artifact = Artifact.from_dict(a, trace_id=trace_id_meta, conversation_id=conversation_id_meta, tool_name=tool_name_meta)
                    normalized_artifacts.append(artifact.to_dict())
                except Exception as ex:
                    log.debug("_canonical_tool_result: Artifact.from_dict failed (non-fatal) ex=%s", ex, exc_info=True)
                    normalized_artifacts.append(a)
            else:
                normalized_artifacts.append(a)
        artifacts = normalized_artifacts
    else:
        artifacts = None
        # Music service-style: artifact_id + relative_url + conversation_id
        rel = data.get("relative_url")
        artifact_id = data.get("artifact_id")
        if isinstance(rel, str) and rel:
            kind: str = "audio"
            low = rel.lower()
            if any(low.endswith(ext) for ext in (".mp4", ".mov", ".mkv", ".webm")):
                kind = "video"
            elif any(low.endswith(ext) for ext in (".wav", ".mp3", ".flac", ".aac", ".ogg")):
                kind = "audio"
            elif any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
                kind = "image"
            # Use build_artifact to ensure proper structure
            trace_id = data.get("trace_id") or ""
            conversation_id = data.get("conversation_id")
            tool_name = data.get("tool_name") or data.get("tool")
            artifacts = [
                build_artifact(
                    artifact_id=artifact_id or rel,
                    kind=kind,
                    path=rel,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    tool_name=tool_name,
                    view_url=rel,
                )
            ]
        # Film export-style: master_uri / reel_mp4 etc.
        elif isinstance(data.get("master_uri"), str):
            trace_id = data.get("trace_id") or ""
            conversation_id = data.get("conversation_id")
            tool_name = data.get("tool_name") or data.get("tool")
            artifacts = [
                build_artifact(
                    artifact_id="master",
                    kind="video",
                    path=data.get("master_uri"),
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    tool_name=tool_name,
                    view_url=data.get("master_uri"),
                )
            ]
        elif isinstance(data.get("reel_mp4"), str):
            trace_id = data.get("trace_id") or ""
            conversation_id = data.get("conversation_id")
            tool_name = data.get("tool_name") or data.get("tool")
            artifacts = [
                build_artifact(
                    artifact_id="reel",
                    kind="video",
                    path=data.get("reel_mp4"),
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    tool_name=tool_name,
                    view_url=data.get("reel_mp4"),
                )
            ]

    out: Dict[str, Any] = {"ids": external_ids, "meta": meta}  # ids contains external API identifiers

    # Preserve rich artifact/QA blocks for media tools (image, video, audio, film, tts, music).
    if isinstance(artifacts, list):
        out["artifacts"] = artifacts
    qa = data.get("qa")
    if isinstance(qa, dict):
        out["qa"] = qa
    
    # ALWAYS preserve errors from the original response - be transparent end-to-end
    # Errors can coexist with successful results (e.g., partial failures, warnings, etc.)
    original_error = x.get("error") if isinstance(x, dict) else None
    if isinstance(original_error, dict) and original_error:
        out["error"] = original_error
    # Also preserve status if present
    original_status = x.get("status") if isinstance(x, dict) else None
    if isinstance(original_status, int):
        out["status"] = original_status

    return out

@app.post("/execute_plan")
async def execute_plan(body: Dict[str, Any]):
    # Single-exit discipline: build `body_env` and return once at the bottom.
    body_env = None

    # Accept either {plan: {...}} or a raw plan object
    plan_obj = body.get("plan") if isinstance(body, dict) and isinstance(body.get("plan"), dict) else (body if isinstance(body, dict) else None)
    trace_id = str(body.get("trace_id") or "").strip() if isinstance(body, dict) and isinstance(body.get("trace_id"), (str, int)) else ""
    conversation_id = str(body.get("conversation_id") or "").strip() if isinstance(body, dict) and isinstance(body.get("conversation_id"), (str, int)) else ""

    if not isinstance(plan_obj, dict):
        logging.error(
            f"[executor] invalid_plan_body trace_id={trace_id} conversation_id={conversation_id} body_type={type(body).__name__} body_keys={sorted(list(body.keys())) if isinstance(body, dict) else []}"
        )
        err_env = {
            "code": "invalid_plan_body",
            "message": "missing_or_invalid_plan",
            "status": 400,
            "details": {
                "reason": "missing_or_invalid_plan",
                "body_type": type(body).__name__,
                "body_keys": sorted(list(body.keys())) if isinstance(body, dict) else [],
            },
            "trace_id": trace_id,
            "conversation_id": conversation_id,
        }
        body_env = _build_executor_envelope(trace_id=trace_id, conversation_id=conversation_id, ok=False, step_outputs_by_step_id={}, error=err_env)
    else:
        raw_steps = plan_obj.get("plan") or []
        if not isinstance(raw_steps, list):
            logging.error(
                f"[executor] invalid_plan_type trace_id={trace_id} conversation_id={conversation_id} plan_type={type(raw_steps).__name__} plan_repr={repr(raw_steps)[:500]}"
            )
            err_env = {
                "code": "invalid_plan_type",
                "message": "plan.plan must be a list",
                "status": 422,
                "details": {
                    "reason": "plan.plan must be a list",
                    "plan_type": type(raw_steps).__name__,
                    "plan_repr": repr(raw_steps)[:500],
                },
                "trace_id": trace_id,
                "conversation_id": conversation_id,
            }
            body_env = _build_executor_envelope(trace_id=trace_id, conversation_id=conversation_id, ok=False, step_outputs_by_step_id={}, error=err_env)
        else:
            # Best-effort execution: never abort on the first failing step.
            # We normalize the plan into `run_steps` format and let it continue
            # running any runnable steps, collecting per-step errors.
            trace_id = str(plan_obj.get("trace_id") or trace_id or "").strip()
            conversation_id = str(plan_obj.get("conversation_id") or conversation_id or "").strip()
            steps: list[dict] = []
            for step in raw_steps:
                if not isinstance(step, dict):
                    continue
                input_keys_required = []
                if "input_keys_required" in step and isinstance(step["input_keys_required"], list):
                    input_keys_required = step["input_keys_required"]
                step_obj: Dict[str, Any] = {
                    "step_id": step.get("step_id"),
                    "tool": (step.get("tool") or "").strip(),
                    # Accept canonical inputs, but also support "args" for compat.
                    "inputs": (step.get("inputs") if isinstance(step.get("inputs"), dict) else (step.get("args") if isinstance(step.get("args"), dict) else {})),
                    "input_keys_required": list(input_keys_required),
                }
                steps.append(step_obj)

            # If the plan is empty, still return a valid envelope.
            step_outputs_by_step_id = await run_steps(trace_id, conversation_id, steps)
            body_env = _build_executor_envelope(trace_id=trace_id, conversation_id=conversation_id, ok=True, step_outputs_by_step_id=step_outputs_by_step_id, error=None)

    return JSONResponse(body_env or {"ok": False, "error": {"code": "internal_error", "message": "executor envelope missing"}}, status_code=200)


async def run_steps(trace_id: str, conversation_id: str, steps: list[dict]):
    # Normalize steps: ids, required inputs, inputs
    norm_steps: Dict[str, Dict[str, Any]] = {}
    used_ids = set()
    auto_idx = 1
    for step in steps:
        if not isinstance(step, dict):
            continue
        # Canonical identifier is step_id (required).
        step_id = (step.get("step_id") or "").strip()
        if not step_id:
            while True:
                step_id = f"step_{auto_idx}"
                auto_idx += 1
                if step_id not in used_ids:
                    break
        if step_id in used_ids:
            k = 2
            base = step_id
            while f"{base}_{k}" in used_ids:
                k += 1
            step_id = f"{base}_{k}"
        used_ids.add(step_id)
        # Accept both "inputs" (canonical) and "args" (compat with orchestrator payload)
        raw_inputs = step.get("inputs")
        if raw_inputs is None:
            raw_inputs = step.get("args")
        if raw_inputs is None:
            raw_inputs = {}
        input_keys_required = []
        if "input_keys_required" in step and isinstance(step["input_keys_required"], list):
            input_keys_required = step["input_keys_required"]
        step_obj = {
            "step_id": step_id,
            "tool": (step.get("tool") or "").strip(),
            "inputs": dict(raw_inputs or {}),
            "input_keys_required": list(input_keys_required),
        }
        norm_steps[step_id] = step_obj

    dependencies: Dict[str, set[str]] = {
        step_id: set((norm_steps[step_id].get("input_keys_required") or [])) for step_id in norm_steps
    }
    trace_id = str(trace_id or "").strip() if isinstance(trace_id, str) else ""
    conversation_id = str(conversation_id or "").strip() if isinstance(conversation_id, str) else ""
    step_outputs_by_step_id: Dict[str, Any] = {}


    def merge_inputs(step: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(step.get("inputs") or {})
        for required_step_id in (step.get("input_keys_required") or []):
            produced_step = step_outputs_by_step_id.get(required_step_id)
            if not isinstance(produced_step, dict):
                continue
            # produced entries are step wrappers: {"tool_name": tool, "result": {...}}.
            # For dependency injection, merge the *result* payload only (best-effort),
            # so downstream steps see the same keys they would have seen in older
            # executor variants that stored raw result dicts in `step_outputs_by_step_id`.
            produced_step_result = produced_step.get("result")
            if isinstance(produced_step_result, dict):
                merged.update(produced_step_result)
        merged["trace_id"] = trace_id
        merged["conversation_id"] = conversation_id
        step_id_local = (step.get("step_id") or "").strip()
        if step_id_local:
            merged["step_id"] = step_id_local
        return merged

    pending = set(norm_steps.keys())
    logging.info("[executor] steps_start trace_id=%s steps=%d", trace_id, len(pending))
    _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
        "t": int(time.time()*1000),
        "event": "exec_plan_start",
        "trace_id": trace_id,
        "steps": len(pending),
    }, expected={})

    while pending:
        satisfied = set(step_outputs_by_step_id.keys())
        runnable = [step_id for step_id in pending if dependencies[step_id].issubset(satisfied)]
        if not runnable:
            # Deadlock or missing dependency: synthesize a tool-style error for the whole plan.
            logging.error(
                f"[executor] deadlock_or_missing trace_id={trace_id} conversation_id={conversation_id} pending={sorted(list(pending))} runnable={sorted(list(runnable))} norm_steps_keys={sorted(list(norm_steps.keys()))}"
            )
            step_outputs_by_step_id["__plan__"] = {
                "tool_name": "executor",
                "result": {
                    "ids": {},
                    "meta": {},
                    "error": {
                        "code": "deadlock_or_missing",
                        "message": "executor detected a dependency deadlock or missing dependency",
                        "trace_id": trace_id,
                        "conversation_id": conversation_id,
                        "status": 422,
                        "details": {
                            "pending": sorted(list(pending)),
                            "runnable": sorted(list(runnable)),
                            "norm_steps_keys": sorted(list(norm_steps.keys())),
                        },
                    },
                    "status": 422,
                },
            }
            break
        batch_tools = [{"step_id": step_id, "tool": norm_steps[step_id].get("tool")} for step_id in runnable]
        logging.info("[executor] batch_start trace_id=%s runnable=%s", trace_id, batch_tools)
        _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
            "t": int(time.time()*1000),
            "event": "exec_batch_start",
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "items": batch_tools,
        }, expected={})
        for step_id in runnable:
            step = norm_steps[step_id]
            tool_name = (step.get("tool") or "").strip()
            if not tool_name:
                logging.error(
                    f"[executor] missing_tool step_id={step_id!r} trace_id={trace_id} conversation_id={conversation_id} step_keys={sorted(list(step.keys())) if isinstance(step, dict) else []}"
                )
                step_outputs_by_step_id[step_id] = {
                    "tool_name": "executor",
                    "result": {
                        "ids": {},
                        "meta": {},
                        "error": {
                            "code": "missing_tool",
                            "message": f"missing tool for step_id {step_id}",
                            "trace_id": trace_id,
                            "conversation_id": conversation_id,
                            "status": 422,
                            "details": {
                                "step_keys": sorted(list(step.keys())) if isinstance(step, dict) else [],
                                "step_repr": repr(step)[:500] if isinstance(step, dict) else str(step)[:500],
                            },
                        },
                        "status": 422,
                    },
                }
                continue
            args = merge_inputs(step)
            t0 = time.time()
            step_start_payload = {
                "t": int(time.time() * 1000),
                "event": "exec_step_start",
                "tool": tool_name,
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "step_id": step_id,
            }
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", step_start_payload, expected={})
            res: Any = None
            ok: bool = False  # Initialize ok to False; will be determined based on actual results
            try:
                res = await utc_run_tool(trace_id, conversation_id, step_id, tool_name, args)
            except Exception as ex:
                exc_tb = traceback.format_exc()
                logging.error(
                    f"[executor] utc_run_tool exception step_id={step_id!r} tool={tool_name!r} trace_id={trace_id} conversation_id={conversation_id} exception_type={type(ex).__name__} exception={ex!r}",
                    exc_info=True
                )
                step_outputs_by_step_id[step_id] = {
                    "tool_name": tool_name,
                    "result": {
                        "ids": {},
                        "meta": {},
                        "error": {
                            "code": "utc_exception",
                            "message": f"utc_run_tool raised: {ex}",
                            "trace_id": trace_id,
                            "conversation_id": conversation_id,
                            "status": 500,
                            "details": {
                                "tool": tool_name,
                                "step_id": step_id,
                                "exception": str(ex),
                                "exception_type": type(ex).__name__,
                                "exception_repr": repr(ex),
                            },
                            "traceback": exc_tb,
                            "stack": exc_tb,
                        },
                        "status": 500,
                        "trace_id": trace_id,
                        "conversation_id": conversation_id,
                    },
                }
                # Still emit end events below; continue to next step.
                # (Do not raise.)
                res = {}
                # Exception occurred, so ok = False
                ok = False
            # Guard: utc_run_tool must never return None/non-dict; if it does, wrap in a result block
            # that matches existing executor result structure (ids/meta/error/status).
            if not isinstance(res, dict):
                logging.error(
                    f"[executor] non_dict_result step_id={step_id!r} tool={tool_name!r} trace_id={trace_id} conversation_id={conversation_id} result_type={type(res).__name__} result_repr={repr(res)[:500]}"
                )
                step_outputs_by_step_id[step_id] = {
                    "tool_name": tool_name,
                    "result": {
                        "ids": {},
                        "meta": {},
                        "error": {
                            "code": "executor_non_dict_result",
                            "message": f"utc_run_tool returned {type(res).__name__} for tool {tool_name}",
                            "trace_id": trace_id,
                            "conversation_id": conversation_id,
                            "status": 422,
                            "details": {
                                "result_type": type(res).__name__,
                                "result_repr": repr(res)[:1000],
                            },
                        },
                        "status": 422,
                    },
                }
                # No results produced, so ok = False
                ok = False
            else:
                try:
                    # Always canonicalize the result first (this preserves errors from the original response)
                    canonical_result = _canonical_tool_result(res or {})
                    
                    # Check if there's an explicit error in the response - ALWAYS log and preserve it
                    err_obj = res.get("error") if isinstance(res.get("error"), dict) else {}
                    has_error = bool(err_obj and isinstance(err_obj, dict) and (err_obj.get("code") or err_obj.get("message")))
                    
                    # If there's an explicit error, normalize it and ensure it's in the result
                    if has_error:
                        # Tool-layer error envelope (from utc_run_tool and/or orchestrator /tool.run).
                        # Prefer fields from the nested error object so logs always show a concrete
                        # code/message/detail instead of falling back to raw HTTP status only.
                        code = err_obj.get("code") if isinstance(err_obj, dict) else None
                        msg = err_obj.get("message") if isinstance(err_obj, dict) else None
                        # Traceback can be in error.traceback, error.stack, or error.details.stack
                        details = err_obj.get("details") if isinstance(err_obj.get("details"), dict) else {}
                        tb = (
                            err_obj.get("traceback") 
                            or err_obj.get("stack") 
                            or details.get("stack")
                            or (details.get("tool_stack") if isinstance(details.get("tool_stack"), (str, bytes)) else None)
                        ) if isinstance(err_obj, dict) else None
                        code = code if isinstance(code, str) and code else "tool_error"
                        msg = msg if isinstance(msg, str) else ""
                        tb = tb if isinstance(tb, (str, bytes)) else ""
                        if tb:
                            logging.error(tb if isinstance(tb, str) else str(tb))
                        # Capture a compact view of the executor/tool env for debugging.
                        env_fields: Dict[str, Any] = {}
                        for k in ("code", "message", "detail", "_http_status", "status"):
                            if k in res:
                                env_fields[k] = res[k]
                        # Also record the shape of the nested error payload so we can see where
                        # fields actually live when debugging.
                        if isinstance(err_obj, dict):
                            env_fields["error_keys"] = sorted(err_obj.keys())
                        # Log using the normalized error object fields - ALWAYS log errors for transparency
                        err_detail = None
                        if isinstance(err_obj, dict):
                            err_detail = err_obj.get("details") or err_obj.get("detail")
                        logging.warning(
                            f"[executor] step_id={step_id!r} tool={tool_name!r} ERROR (may have partial results) code={code!r} msg={msg!r} detail={err_detail!r} env={json.dumps(env_fields, sort_keys=True)!r}"
                        )
                        # Normalize the error payload so it always carries a meaningful message.
                        error_dict: Dict[str, Any] = err_obj if isinstance(err_obj, dict) else {}
                        if not isinstance(error_dict, dict):
                            error_dict = {}
                        # Prefer status from the tool error payload when present.
                        status = error_dict.get("status") if isinstance(error_dict.get("status"), int) else (
                            res.get("status") if isinstance(res.get("status"), int) else 422
                        )
                        if "code" not in error_dict and code:
                            error_dict["code"] = code
                        if not error_dict.get("message"):
                            error_dict["message"] = msg or f"{tool_name} had error with status {status}"
                        if "status" not in error_dict:
                            error_dict["status"] = status
                        if tb and not any(k in error_dict for k in ("traceback", "stack", "stacktrace")):
                            error_dict["traceback"] = tb
                        # ALWAYS inject error into the canonical result (even if ok = True)
                        if isinstance(canonical_result, dict):
                            canonical_result["error"] = error_dict
                            if "status" not in canonical_result:
                                canonical_result["status"] = status
                    
                    # Store the result (with errors preserved if present)
                    step_outputs_by_step_id[step_id] = {"tool_name": tool_name, "result": canonical_result}
                    
                    # Determine ok based on whether the tool actually produced useful results
                    # (artifacts, ids, or meaningful data), NOT based on errors
                    final_result = canonical_result
                    if not isinstance(final_result, dict):
                        ok = False
                    else:
                        # Check for artifacts (most reliable indicator of success)
                        artifacts = final_result.get("artifacts")
                        has_artifacts = isinstance(artifacts, list) and len(artifacts) > 0
                        
                        # Check for ids (external identifiers indicate work was done)
                        ids = final_result.get("ids")
                        has_ids = isinstance(ids, dict) and len(ids) > 0
                        
                        # Check for meaningful meta data
                        meta = final_result.get("meta")
                        has_meaningful_meta = isinstance(meta, dict) and len(meta) > 0
                        
                        # Check for other result fields that indicate work was done
                        has_result_data = any(
                            final_result.get(k) is not None 
                            for k in ("url", "view_url", "data_url", "relative_url", "master_uri", "artifact_id", "windows", "ids")
                        )
                        
                        # ok = True if tool produced any useful results, regardless of errors
                        ok = has_artifacts or has_ids or has_meaningful_meta or has_result_data
                except Exception as ex:
                    # If processing the result fails, log it and set ok = False
                    exc_tb = traceback.format_exc()
                    logging.error(
                        f"[executor] result processing exception step_id={step_id!r} tool={tool_name!r} trace_id={trace_id} conversation_id={conversation_id} exception_type={type(ex).__name__} exception={ex!r}",
                        exc_info=True
                    )
                    step_outputs_by_step_id[step_id] = {
                        "tool_name": tool_name,
                        "result": {
                            "ids": {},
                            "meta": {},
                            "error": {
                                "code": "result_processing_exception",
                                "message": f"Failed to process tool result: {ex}",
                                "trace_id": trace_id,
                                "conversation_id": conversation_id,
                                "status": 500,
                                "details": {
                                    "exception": str(ex),
                                    "exception_type": type(ex).__name__,
                                    "exception_repr": repr(ex),
                                },
                                "traceback": exc_tb,
                                "stack": exc_tb,
                            },
                            "status": 500,
                        },
                    }
                    ok = False
            
            # Ensure ok is always set (fallback for cases where we didn't enter the else block)
            # This should never happen since ok is initialized to False, but just to be safe
            if step_id not in step_outputs_by_step_id:
                ok = False
            # Extract error from final result for end_payload (always include errors for transparency)
            final_result = step_outputs_by_step_id.get(step_id, {}).get("result", {})
            final_error = final_result.get("error") if isinstance(final_result, dict) else None
            final_traceback = None
            if isinstance(final_error, dict):
                # Traceback can be in error.traceback, error.stack, error.stacktrace, or error.details.stack
                details = final_error.get("details") if isinstance(final_error.get("details"), dict) else {}
                final_traceback = (
                    final_error.get("traceback") 
                    or final_error.get("stack") 
                    or final_error.get("stacktrace")
                    or details.get("stack")
                    or (details.get("tool_stack") if isinstance(details.get("tool_stack"), (str, bytes)) else None)
                )
            
            end_payload = {
                "t": int(time.time()*1000),
                "event": "end",
                "tool": tool_name,
                "ok": bool(ok),
                "duration_ms": int((time.time() - t0) * 1000.0),
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "step_id": step_id,
                # IMPORTANT: always include errors for transparency (even if ok = True)
                # Downstream (AI, UI) needs to see all errors to make informed decisions
                "error": final_error if isinstance(final_error, dict) else None,
                "traceback": final_traceback if isinstance(final_traceback, (str, bytes)) else None,
                "summary": (_distill_summary(step_outputs_by_step_id.get(step_id)) if ok else None),
            }
            if not ok:
                step_obj = step_outputs_by_step_id.get(step_id) if isinstance(step_outputs_by_step_id.get(step_id), dict) else {}
                res_obj = step_obj.get("result") if isinstance(step_obj.get("result"), dict) else {}
                err_obj = res_obj.get("error") if isinstance(res_obj.get("error"), dict) else None
                if err_obj is None:
                    # Fall back to a minimal structured error envelope.
                    err_obj = {"code": "tool_error", "message": f"{tool_name} failed", "status": int(res_obj.get("status") or 422)}
                end_payload["error"] = err_obj
                tb = err_obj.get("traceback") if isinstance(err_obj, dict) else None
                if not isinstance(tb, (str, bytes)) or not tb:
                    tb = err_obj.get("stack") if isinstance(err_obj, dict) else None
                if isinstance(tb, (str, bytes)) and tb:
                    end_payload["traceback"] = tb
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", end_payload, expected={})

            step_finish_payload = {
                "t": int(time.time()*1000),
                "event": "exec_step_finish",
                "tool": tool_name,
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "step_id": step_id,
                "ok": bool(ok),
            }
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", step_finish_payload, expected={})
        for step_id in runnable:
            pending.remove(step_id)
        logging.info("[executor] batch_finish trace_id=%s runnable=%s", trace_id, batch_tools)
        _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
            "t": int(time.time()*1000),
            "event": "exec_batch_finish",
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "items": batch_tools,
        }, expected={})
    logging.info("[executor] steps_finish trace_id=%s produced_keys=%s", trace_id, sorted(list(step_outputs_by_step_id.keys())))
    _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
        "t": int(time.time()*1000),
        "event": "exec_plan_finish",
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "produced_keys": sorted(list(step_outputs_by_step_id.keys())),
    }, expected={})
    return step_outputs_by_step_id


@app.post("/execute")
async def execute_http(body: Dict[str, Any]):
    trace_id = str(body.get("trace_id") or "").strip() if isinstance(body.get("trace_id"), (str, int)) else ""
    conversation_id = str(body.get("conversation_id") or "").strip() if isinstance(body.get("conversation_id"), (str, int)) else ""
    steps = body.get("steps") if isinstance(body.get("steps"), list) else None

    # Single unified return: always build a full envelope and return once.
    step_outputs_by_step_id: Dict[str, Any] = {}
    error_obj: Dict[str, Any] | None = None

    if not steps:
        # No steps: synthesize an explicit invalid_plan error.
        logging.error(
            "[executor] invalid_plan_no_steps trace_id=%s conversation_id=%s body_keys=%s body_type=%s",
            trace_id, conversation_id, sorted(list(body.keys())) if isinstance(body, dict) else [], type(body).__name__
        )
        step_outputs_by_step_id = {}
        error_obj = {
            "code": "invalid_plan",
            "message": "no steps provided to executor; 'steps' field is required",
            "status": 422,
            "details": {
                "reason": "steps required",
                "body_keys": sorted(list(body.keys())) if isinstance(body, dict) else [],
                "body_type": type(body).__name__,
            },
        }
    else:
        # NEVER break the flow - always handle errors gracefully
        try:
            step_outputs_by_step_id = await run_steps(trace_id, conversation_id, steps)
        except Exception as ex:
            exc_tb = traceback.format_exc()
            logging.error(
                "[executor] run_steps_exception trace_id=%s conversation_id=%s exception_type=%s exception=%r",
                trace_id, conversation_id, type(ex).__name__, ex,
                exc_info=True
            )
            # Return empty step outputs and surface the exception as an error
            step_outputs_by_step_id = {}
            error_obj = {
                "code": "run_steps_exception",
                "message": f"run_steps raised exception: {ex}",
                "status": 500,
                "details": {
                    "exception": str(ex),
                    "exception_type": type(ex).__name__,
                    "exception_repr": repr(ex),
                },
                "traceback": exc_tb,
                "stack": exc_tb,
            }

    # Derive overall executor status from per-step results when we actually ran
    # a plan. Any step with a non-200 status is treated as a hard failure and
    # surfaces as a top-level error so callers never see ok:true with
    # step.status != 200.
    if error_obj is None:
        failing: list[tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for step_id, step in step_outputs_by_step_id.items():
            if not isinstance(step, dict):
                continue
            res = step.get("result")
            if not isinstance(res, dict):
                continue
            st = res.get("status")
            if isinstance(st, int) and st != 200:
                failing.append((str(step_id), step, res))

        if failing:
            # Surface the first failing step as the canonical executor error.
            first_sid, first_step, first_res = failing[0]
            err_obj = first_res.get("error") if isinstance(first_res.get("error"), dict) else {}
            tool_name = first_step.get("tool_name") if isinstance(first_step.get("tool_name"), str) else first_step.get("tool")
            code = err_obj.get("code") if isinstance(err_obj, dict) else None
            msg = err_obj.get("message") if isinstance(err_obj, dict) else None
            status = err_obj.get("status") if isinstance(err_obj, dict) else first_res.get("status")
            # Preserve the ORIGINAL tool traceback/stack when available.
            traceback_text: str | bytes | None = None
            if isinstance(err_obj, dict):
                for k in ("traceback", "stack", "stacktrace"):
                    v = err_obj.get(k)
                    if isinstance(v, (str, bytes)) and v:
                        traceback_text = v
                        break
            if traceback_text is None and isinstance(first_res, dict):
                for k in ("traceback", "stack", "stacktrace"):
                    v = first_res.get(k)
                    if isinstance(v, (str, bytes)) and v:
                        traceback_text = v
                        break
            code = str(code) if isinstance(code, str) and code else "executor_step_failed"
            # Human-readable message: prefer the tool's own message, fall back to concise summary.
            human_msg = (
                str(msg).strip()
                if isinstance(msg, str) and msg and msg.strip()
                else "step %s (%s) failed with status %s" % (first_sid, tool_name or 'tool', status)
            )
            stk = traceback_text if isinstance(traceback_text, (str, bytes)) and traceback_text else None
            if not stk:
                # Use traceback.format_stack() which works outside exception handlers
                try:
                    stk = "".join(traceback.format_stack())
                except Exception:
                    try:
                        stk = traceback.format_exc()
                    except Exception:
                        stk = "<stack trace unavailable>"
            # Ensure stk is a string
            if not isinstance(stk, str):
                if isinstance(stk, bytes):
                    try:
                        stk = stk.decode("utf-8", errors="replace")
                    except Exception:
                        stk = "<stack trace unavailable>"
                else:
                    stk = "<stack trace unavailable>"
            status_int = int(status) if isinstance(status, int) else 422
            logging.error(
                "[executor] step_failed trace_id=%s conversation_id=%s step_id=%s tool=%r code=%r message=%r status=%d",
                trace_id, conversation_id, first_sid, tool_name, code, human_msg, status_int
            )
            error_obj = {
                "code": code,
                "message": human_msg,
                "status": status_int,
                "tool": tool_name,
                "step": first_sid,
                "details": {
                    "summary": human_msg,
                    "tool_error": err_obj or {},
                },
                # Full underlying traceback/stack from the failing tool is still present here.
                "traceback": stk,
                "stack": stk,
                # Also expose the raw step result so nothing about the failure is hidden.
                "tool_result": first_res,
            }

    body = _build_executor_envelope(trace_id=trace_id, conversation_id=conversation_id, ok=(error_obj is None), step_outputs_by_step_id=step_outputs_by_step_id, error=error_obj)
    # Always return HTTP 200; callers MUST inspect error.status / ok to
    # determine failure. This keeps the transport-layer status stable while
    # still surfacing the exact tool failure and full stack trace.
    return JSONResponse(body, status_code=200)

