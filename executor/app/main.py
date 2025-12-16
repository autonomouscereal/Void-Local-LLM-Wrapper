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

import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from executor.utc.utc_runner import utc_run_tool  # moved to top per import policy
from executor.utc.db import ensure_tables
import uuid


WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
EXEC_TIMEOUT_SEC = int(os.getenv("EXEC_TIMEOUT_SEC", "30"))
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
        stream=sys.stdout,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(_log_file, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger("executor.logging").info("executor logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("executor.logging").warning("executor file logging disabled: %s", _ex, exc_info=True)
log = logging.getLogger("executor")


def _stack_str() -> str:
    """Return a best-effort string representation of the current call stack."""
    return "".join(traceback.format_stack())


@app.on_event("startup")
async def _startup():
    # Let startup failures surface normally; no swallowing.
    await ensure_tables()
    logging.info("[executor] UTC tables ensured")

def within_workspace(path: str) -> str:
    full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    # Guard against path traversal; callers should surface this as an error envelope.
    if not full.startswith(os.path.abspath(WORKSPACE_DIR)):
        return os.path.abspath(WORKSPACE_DIR)
    return full


def run_subprocess(cmd: list[str], cwd: Optional[str] = None, timeout: int = EXEC_TIMEOUT_SEC) -> Dict[str, Any]:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd or WORKSPACE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return {"returncode": proc.returncode, "stdout": out, "stderr": err}


def _banned_reason_from_code(code: str) -> Optional[str]:
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


def _banned_reason_from_shell(cmd: str) -> Optional[str]:
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


def _post_json(url: str, payload: Dict[str, Any], expected: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        parser = JSONParser()
        schema = expected if expected is not None else {}
        return parser.parse(raw, schema)


def _distill_summary(result_obj: Any) -> Dict[str, Any]:
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
    request_id: str,
    ok: bool,
    produced: Dict[str, Any] | None,
    error: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Canonical executor envelope builder.

    This mirrors the orchestrator ToolEnvelope shape:
    {
      "schema_version": 1,
      "request_id": "...",
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
        for step in (produced or {}).values():
            if not isinstance(step, dict):
                continue
            res = step.get("result")
            if not isinstance(res, dict):
                continue
            st = res.get("status")
            if isinstance(st, int) and st != 200:
                final_ok = False
                break
    except Exception:
        # Never let envelope construction fail; fall back to the provided ok hint.
        final_ok = bool(ok)
    return {
        "schema_version": 1,
        "request_id": request_id,
        "ok": final_ok,
        "result": {"produced": produced or {}},
        "error": error,
    }


def _canonical_tool_result(x: Dict[str, Any] | Any) -> Dict[str, Any]:
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

    ids = data.get("ids") if isinstance(data.get("ids"), dict) else {}
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
    # music/tts/film tools often surface cid/project_id at the top level.
    if "cid" in data and "cid" not in meta and isinstance(data.get("cid"), str):
        meta["cid"] = data.get("cid")
    if "project_id" in data and "project_id" not in ids and isinstance(data.get("project_id"), str):
        ids["project_id"] = data.get("project_id")

    # Heuristic normalization for single-artifact tools that don't emit an explicit
    # artifacts list (e.g. services/music, some audio/video helpers).
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = None
        # Music service-style: artifact_id + relative_url + cid
        rel = data.get("relative_url")
        art_id = data.get("artifact_id") or data.get("id")
        if isinstance(rel, str) and rel:
            kind: str = "audio"
            low = rel.lower()
            if any(low.endswith(ext) for ext in (".mp4", ".mov", ".mkv", ".webm")):
                kind = "video"
            elif any(low.endswith(ext) for ext in (".wav", ".mp3", ".flac", ".aac", ".ogg")):
                kind = "audio"
            elif any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
                kind = "image"
            artifacts = [
                {
                    "id": art_id or rel,
                    "kind": kind,
                    "path": rel,
                    "view_url": rel,
                }
            ]
        # Film export-style: master_uri / reel_mp4 etc.
        elif isinstance(data.get("master_uri"), str):
            artifacts = [
                {
                    "id": "master",
                    "kind": "video",
                    "path": data.get("master_uri"),
                    "view_url": data.get("master_uri"),
                }
            ]
        elif isinstance(data.get("reel_mp4"), str):
            artifacts = [
                {
                    "id": "reel",
                    "kind": "video",
                    "path": data.get("reel_mp4"),
                    "view_url": data.get("reel_mp4"),
                }
            ]

    out: Dict[str, Any] = {"ids": ids, "meta": meta}

    # Preserve rich artifact/QA blocks for media tools (image, video, audio, film, tts, music).
    if isinstance(artifacts, list):
        out["artifacts"] = artifacts
    qa = data.get("qa")
    if isinstance(qa, dict):
        out["qa"] = qa

    return out

@app.post("/execute_plan")
async def execute_plan(body: Dict[str, Any]):
    # Accept either {plan: {...}} or a raw plan object
    plan_obj = body.get("plan") if isinstance(body, dict) and isinstance(body.get("plan"), dict) else (body if isinstance(body, dict) else None)
    if not isinstance(plan_obj, dict):
        # Use canonical executor envelope even for invalid bodies so callers
        # never have to special-case this endpoint.
        rid = str(uuid.uuid4().hex)
        stk = _stack_str()
        err_env = {
            "code": "invalid_plan_body",
            "message": stk,
            "status": 400,
            "details": {"reason": "missing_or_invalid_plan"},
            "traceback": stk,
            "stack": stk,
        }
        body_env = _build_executor_envelope(rid, False, {}, err_env)
        return JSONResponse(body_env, status_code=200)

    raw_steps = plan_obj.get("plan") or []
    if not isinstance(raw_steps, list):
        rid = str(plan_obj.get("request_id") or uuid.uuid4().hex)
        stk = _stack_str()
        err_env = {
            "code": "invalid_plan_type",
            "message": stk,
            "status": 422,
            "details": {"reason": "plan.plan must be a list"},
            "traceback": stk,
            "stack": stk,
        }
        body_env = _build_executor_envelope(rid, False, {}, err_env)
        return JSONResponse(body_env, status_code=200)

    # request_id (rid) and trace_id are distinct identifiers; never reuse one as the other.
    request_id_val = (
        str(plan_obj.get("request_id")).strip()
        if isinstance(plan_obj.get("request_id"), str) and str(plan_obj.get("request_id")).strip()
        else uuid.uuid4().hex
    )
    trace_id_val = (
        str(plan_obj.get("trace_id")).strip()
        if isinstance(plan_obj.get("trace_id"), str) and str(plan_obj.get("trace_id")).strip()
        else uuid.uuid4().hex
    )

    # Normalize steps: ids, needs, inputs
    norm_steps: Dict[str, Dict[str, Any]] = {}
    used_ids = set()
    auto_idx = 1
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        sid = (s.get("id") or "").strip()
        if not sid:
            while True:
                sid = f"s{auto_idx}"
                auto_idx += 1
                if sid not in used_ids:
                    break
        if sid in used_ids:
            k = 2
            base = sid
            while f"{base}_{k}" in used_ids:
                k += 1
            sid = f"{base}_{k}"
        used_ids.add(sid)
        step = {
            "id": sid,
            "tool": (s.get("tool") or "").strip(),
            "inputs": dict(s.get("inputs") or {}),
            "needs": list(s.get("needs") or []),
        }
        norm_steps[sid] = step

    deps: Dict[str, set[str]] = {sid: set((norm_steps[sid].get("needs") or [])) for sid in norm_steps}
    produced: Dict[str, Any] = {}

    def _merge_inputs(step: Dict[str, Any]) -> Dict[str, Any]:
        inputs = dict(step.get("inputs") or {})
        for need in (step.get("needs") or []):
            if need in produced and isinstance(produced[need], dict):
                inputs.update(produced[need])
        # Attach trace metadata for orchestrator tracing; overwrite any caller-
        # supplied values so ids remain fully server-controlled.
        if trace_id_val:
            inputs["trace_id"] = trace_id_val
        sid_local = (step.get("id") or "").strip()
        if sid_local:
            inputs["step_id"] = sid_local
        return inputs

    pending = set(norm_steps.keys())
    logging.info(f"[executor] plan_start trace_id={trace_id_val} steps={len(pending)}")
    while pending:
        satisfied = set(produced.keys())
        runnable = [sid for sid in pending if deps[sid].issubset(satisfied)]
        if not runnable:
            rid = str(request_id_val)
            stk = _stack_str()
            err_env = {
                "code": "deadlock_or_missing",
                "message": stk,
                "status": 422,
                "details": {"pending": sorted(list(pending))},
                "traceback": stk,
                "stack": stk,
            }
            body_env = _build_executor_envelope(rid, False, produced, err_env)
            return JSONResponse(body_env, status_code=200)
        batch_tools = [{"step_id": sid, "tool": norm_steps[sid].get("tool")} for sid in runnable]
        logging.info(f"[executor] batch_start trace_id={trace_id_val} runnable={batch_tools}")
        for sid in runnable:
            step = norm_steps[sid]
            tool_name = (step.get("tool") or "").strip()
            if not tool_name:
                rid = str(request_id_val)
                stk = _stack_str()
                err_env = {
                    "code": "missing_tool",
                    "message": stk,
                    "status": 422,
                    "details": {"step": sid},
                    "traceback": stk,
                    "stack": stk,
                }
                body_env = _build_executor_envelope(rid, False, produced, err_env)
                return JSONResponse(body_env, status_code=200)
            args = _merge_inputs(step)
            t0 = time.time()
            # Use UTC runner directly; any exceptions bubble to the global exception handler.
            res = await utc_run_tool(trace_id_val, sid, tool_name, args)
            ok = False
            if isinstance(res, dict) and bool(res.get("ok")) is True and res.get("result") is not None:
                produced[sid] = res.get("result")
                ok = True
            elif isinstance(res, dict) and bool(res.get("ok")) is False:
                # Canonicalize tool error into the same envelope shape used by
                # /execute, including full stack trace in the message field.
                err_obj = res.get("error") if isinstance(res.get("error"), dict) else {}
                code_val = err_obj.get("code") if isinstance(err_obj, dict) else None
                msg_val = err_obj.get("message") if isinstance(err_obj, dict) else None
                tb_val = err_obj.get("traceback") if isinstance(err_obj, dict) else None
                code = code_val if isinstance(code_val, str) and code_val else "tool_error"
                human_msg = msg_val if isinstance(msg_val, str) else ""
                tb = tb_val if isinstance(tb_val, (str, bytes)) else _stack_str()
                if tb:
                    logging.error(tb if isinstance(tb, str) else str(tb))
                err_env = {
                    "code": code,
                    "message": tb,
                    "status": 422,
                    "step": sid,
                    "details": {"summary": human_msg, "attempts": int(max(1, MAX_TOOL_ATTEMPTS))},
                    "traceback": tb,
                    "stack": tb,
                }
                body = _build_executor_envelope(
                    str(request_id_val or "executor-plan"),
                    False,
                    produced,
                    err_env,
                )
                return JSONResponse(body, status_code=200)
            else:
                produced[sid] = (res or {})
                ok = True
            # Append end event to orchestrator tool logs is skipped to avoid network exceptions
            # Distilled executor step finish skipped to avoid network exceptions
        for sid in runnable:
            pending.remove(sid)
        logging.info(f"[executor] batch_finish trace_id={trace_id_val} runnable={batch_tools}")
    logging.info(f"[executor] plan_finish trace_id={trace_id_val} produced_keys={sorted(list(produced.keys()))}")
    # For legacy execute_plan, still return a canonical executor envelope so
    # callers can rely on a single shape. No error object here because all
    # steps completed without tool_error.
    body = _build_executor_envelope(str(request_id_val), True, produced, None)
    return JSONResponse(body, status_code=200)


async def run_steps(trace_id: str, request_id: str, steps: list[dict]) -> Dict[str, Any]:
    # Normalize steps: ids, needs, inputs
    norm_steps: Dict[str, Dict[str, Any]] = {}
    used_ids = set()
    auto_idx = 1
    for s in steps:
        if not isinstance(s, dict):
            continue
        sid = (s.get("id") or "").strip()
        if not sid:
            while True:
                sid = f"s{auto_idx}"
                auto_idx += 1
                if sid not in used_ids:
                    break
        if sid in used_ids:
            k = 2
            base = sid
            while f"{base}_{k}" in used_ids:
                k += 1
            sid = f"{base}_{k}"
        used_ids.add(sid)
        # Accept both "inputs" (canonical) and "args" (compat with orchestrator payload)
        raw_inputs = s.get("inputs")
        if raw_inputs is None:
            raw_inputs = s.get("args")
        if raw_inputs is None:
            raw_inputs = {}
        step = {
            "id": sid,
            "tool": (s.get("tool") or "").strip(),
            "inputs": dict(raw_inputs or {}),
            "needs": list(s.get("needs") or []),
        }
        norm_steps[sid] = step

    dependencies: Dict[str, set[str]] = {sid: set((norm_steps[sid].get("needs") or [])) for sid in norm_steps}
    produced: Dict[str, Any] = {}
    # Correlation fallback: if trace_id is missing/falsy, fall back to request_id.
    # This preserves downstream trace correlation and debugging.
    trace_corr = (str(trace_id).strip() if isinstance(trace_id, str) and str(trace_id).strip() else "") or (
        str(request_id).strip() if isinstance(request_id, str) and str(request_id).strip() else ""
    )

    def merge_inputs(step: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(step.get("inputs") or {})
        for need in (step.get("needs") or []):
            if need in produced and isinstance(produced[need], dict):
                merged.update(produced[need])
        # Trace id must be stable and come from the /execute payload.
        # CID should be taken from step args when present; do not fabricate one.
        if trace_corr:
            merged["trace_id"] = trace_corr
        sid_local = (step.get("id") or "").strip()
        if sid_local:
            merged["step_id"] = sid_local
        return merged

    pending = set(norm_steps.keys())
    logging.info(f"[executor] steps_start trace_id={trace_corr} steps={len(pending)}")
    _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
        "t": int(time.time()*1000),
        "event": "exec_plan_start",
        "trace_id": trace_corr,
        "steps": len(pending),
    }, expected={})

    while pending:
        satisfied = set(produced.keys())
        runnable = [sid for sid in pending if dependencies[sid].issubset(satisfied)]
        if not runnable:
            # Deadlock or missing dependency: synthesize a tool-style error for the whole plan.
            produced["__plan__"] = {
                "name": "executor",
                "result": {
                    "ids": {},
                    "meta": {},
                    "error": {
                        "code": "deadlock_or_missing",
                        "message": "executor detected a dependency deadlock or missing dependency",
                        "stack": _stack_str(),
                        "trace_id": trace_corr,
                    },
                    "status": 422,
                },
            }
            break
        batch_tools = [{"step_id": sid, "tool": norm_steps[sid].get("tool")} for sid in runnable]
        logging.info(f"[executor] batch_start trace_id={trace_corr} runnable={batch_tools}")
        _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
            "t": int(time.time()*1000),
            "event": "exec_batch_start",
            "trace_id": trace_corr,
            "items": batch_tools,
        }, expected={})
        for sid in runnable:
            step = norm_steps[sid]
            tool_name = (step.get("tool") or "").strip()
            if not tool_name:
                produced[sid] = {
                    "name": "executor",
                    "result": {
                        "ids": {},
                        "meta": {},
                        "error": {
                            "code": "missing_tool",
                            "message": f"missing tool for step {sid}",
                            "stack": _stack_str(),
                            "trace_id": trace_corr,
                        },
                        "status": 422,
                    },
                }
                continue
            args = merge_inputs(step)
            cid_val = args.get("cid") if isinstance(args, dict) else None
            cid_val = cid_val.strip() if isinstance(cid_val, str) else None
            t0 = time.time()
            step_start_payload = {
                "t": int(time.time() * 1000),
                "event": "exec_step_start",
                "tool": tool_name,
                "trace_id": trace_corr,
                "step_id": sid,
            }
            if cid_val:
                step_start_payload["cid"] = cid_val
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", step_start_payload, expected={})
            res = await utc_run_tool(trace_corr, sid, tool_name, args)
            ok = False
            # Guard: utc_run_tool must never return None/non-dict; if it does, wrap in a result block
            # that matches existing executor result structure (ids/meta/error/status), with a stack.
            if not isinstance(res, dict):
                produced[sid] = {
                    "name": tool_name,
                    "result": {
                        "ids": {},
                        "meta": {},
                        "error": {
                            "code": "executor_non_dict_result",
                            "message": f"utc_run_tool returned {type(res).__name__} for tool {tool_name}",
                            "stack": _stack_str(),
                            "trace_id": trace_corr,
                        },
                        "status": 422,
                    },
                }
            elif bool(res.get("ok")) is True and res.get("result") is not None:
                produced[sid] = {"name": tool_name, "result": _canonical_tool_result(res)}
                ok = True
            elif bool(res.get("ok")) is False:
                # Tool-layer error envelope (from utc_run_tool and/or orchestrator /tool.run).
                # Prefer fields from the nested error object so logs always show a concrete
                # code/message/detail instead of falling back to raw HTTP status only.
                err_obj = res.get("error") if isinstance(res.get("error"), dict) else {}
                code_val = err_obj.get("code") if isinstance(err_obj, dict) else None
                msg_val = err_obj.get("message") if isinstance(err_obj, dict) else None
                tb_val = err_obj.get("traceback") if isinstance(err_obj, dict) else None
                code = code_val if isinstance(code_val, str) and code_val else "tool_error"
                msg = msg_val if isinstance(msg_val, str) else ""
                tb = tb_val if isinstance(tb_val, (str, bytes)) else ""
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
                # Log using the normalized error object fields.
                err_detail = None
                if isinstance(err_obj, dict):
                    err_detail = err_obj.get("details") or err_obj.get("detail")
                logging.error(
                    "[executor] step=%s tool=%s FAILED code=%s msg=%s detail=%s env=%s",
                    sid,
                    tool_name,
                    code,
                    msg,
                    err_detail,
                    json.dumps(env_fields, sort_keys=True),
                )
                # Normalize the error payload so it always carries a meaningful message.
                error_dict: Dict[str, Any] = err_obj if isinstance(err_obj, dict) else {}
                if not isinstance(error_dict, dict):
                    error_dict = {}
                # Prefer status from the tool error payload when present.
                status_val = error_dict.get("status") if isinstance(error_dict.get("status"), int) else (
                    res.get("status") if isinstance(res.get("status"), int) else 422
                )
                if "code" not in error_dict and code:
                    error_dict["code"] = code
                if not error_dict.get("message"):
                    error_dict["message"] = msg or f"{tool_name} failed with status {status_val}"
                if "status" not in error_dict:
                    error_dict["status"] = status_val
                if tb and not any(k in error_dict for k in ("traceback", "stack", "stacktrace")):
                    error_dict["traceback"] = tb
                produced[sid] = {
                    "name": tool_name,
                    "result": {
                        "ids": {},
                        "meta": {},
                        "error": error_dict,
                        "status": status_val,
                        "trace_id": (
                            res.get("trace_id")
                            if isinstance(res.get("trace_id"), str) and str(res.get("trace_id")).strip()
                            else trace_corr
                        ),
                    },
                }
            else:
                produced[sid] = {"name": tool_name, "result": _canonical_tool_result(res or {})}
                ok = True
            end_payload = {
                "t": int(time.time()*1000),
                "event": "end",
                "tool": tool_name,
                "ok": bool(ok),
                "duration_ms": int((time.time() - t0) * 1000.0),
                "trace_id": trace_corr,
                "step_id": sid,
                "error": (None if ok else "tool_error"),
                "traceback": None,
                "summary": (_distill_summary(produced.get(sid)) if ok else None),
            }
            if cid_val:
                end_payload["cid"] = cid_val
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", end_payload, expected={})

            step_finish_payload = {
                "t": int(time.time()*1000),
                "event": "exec_step_finish",
                "tool": tool_name,
                "trace_id": trace_corr,
                "step_id": sid,
                "ok": bool(ok),
            }
            if cid_val:
                step_finish_payload["cid"] = cid_val
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", step_finish_payload, expected={})
        for sid in runnable:
            pending.remove(sid)
        logging.info(f"[executor] batch_finish trace_id={trace_corr} runnable={batch_tools}")
        _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
            "t": int(time.time()*1000),
            "event": "exec_batch_finish",
            "trace_id": trace_corr,
            "items": batch_tools,
        }, expected={})
    logging.info(f"[executor] steps_finish trace_id={trace_corr} produced_keys={sorted(list(produced.keys()))}")
    _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
        "t": int(time.time()*1000),
        "event": "exec_plan_finish",
        "trace_id": trace_corr,
        "produced_keys": sorted(list(produced.keys())),
    }, expected={})
    return produced


@app.post("/execute")
async def execute_http(body: Dict[str, Any]):
    rid = str(body.get("request_id") or uuid.uuid4().hex)
    tid = str(body.get("trace_id") or uuid.uuid4().hex)
    steps = body.get("steps") if isinstance(body.get("steps"), list) else None

    # Single unified return: always build a full envelope and return once.
    produced: Dict[str, Any] = {}
    error_obj: Dict[str, Any] | None = None

    if not steps:
        # No steps: synthesize an explicit invalid_plan error, but keep the
        # human-friendly message separate from the low-level stack trace.
        produced = {}
        stk = _stack_str()
        error_obj = {
            "code": "invalid_plan",
            "message": "no steps provided to executor; 'steps' field is required",
            "status": 422,
            "details": {"reason": "steps required"},
            "traceback": stk,
            "stack": stk,
        }
    else:
        produced = await run_steps(tid, rid, steps)

    # Derive overall executor status from per-step results when we actually ran
    # a plan. Any step with a non-200 status is treated as a hard failure and
    # surfaces as a top-level error so callers never see ok:true with
    # step.status != 200.
    if error_obj is None:
        failing: list[tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for sid, step in produced.items():
            if not isinstance(step, dict):
                continue
            res = step.get("result")
            if not isinstance(res, dict):
                continue
            st = res.get("status")
            if isinstance(st, int) and st != 200:
                failing.append((str(sid), step, res))

        if failing:
            # Surface the first failing step as the canonical executor error.
            first_sid, first_step, first_res = failing[0]
            err_obj = first_res.get("error") if isinstance(first_res.get("error"), dict) else {}
            tool_name = first_step.get("name") if isinstance(first_step.get("name"), str) else first_step.get("tool")
            code_val = err_obj.get("code") if isinstance(err_obj, dict) else None
            msg_val = err_obj.get("message") if isinstance(err_obj, dict) else None
            status_val = err_obj.get("status") if isinstance(err_obj, dict) else first_res.get("status")
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
            code = str(code_val) if isinstance(code_val, str) and code_val else "executor_step_failed"
            # Human-readable message: prefer the tool's own message, fall back to concise summary.
            human_msg = (
                str(msg_val).strip()
                if isinstance(msg_val, str) and msg_val and msg_val.strip()
                else f"step {first_sid} ({tool_name or 'tool'}) failed with status {status_val}"
            )
            stk = (
                traceback_text
                if isinstance(traceback_text, (str, bytes)) and traceback_text
                else _stack_str()
            )
            status_int = int(status_val) if isinstance(status_val, int) else 422
            error_obj = {
                "code": code,
                "message": human_msg,
                "status": status_int,
                "tool": tool_name,
                "step": first_sid,
                "details": {"summary": human_msg, "tool_error": err_obj or {}},
                # Full underlying traceback/stack from the failing tool is still present here.
                "traceback": stk,
                "stack": stk,
                # Also expose the raw step result so nothing about the failure is hidden.
                "tool_result": first_res,
            }

    body = _build_executor_envelope(rid, error_obj is None, produced, error_obj)
    # Always return HTTP 200; callers MUST inspect error.status / ok to
    # determine failure. This keeps the transport-layer status stable while
    # still surfacing the exact tool failure and full stack trace.
    return JSONResponse(body, status_code=200)

