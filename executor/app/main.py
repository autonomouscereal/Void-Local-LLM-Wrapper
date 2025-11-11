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
sys.path.append("/workspace")
from orchestrator.app.json_parser import JSONParser  # use the same hardened JSON parser

import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uuid


WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
EXEC_TIMEOUT_SEC = int(os.getenv("EXEC_TIMEOUT_SEC", "30"))
EXEC_MEMORY_MB = int(os.getenv("EXEC_MEMORY_MB", "2048"))
ALLOW_SHELL = os.getenv("ALLOW_SHELL", "false").lower() == "true"
SHELL_WHITELIST = set([s for s in (os.getenv("SHELL_WHITELIST") or "").split(",") if s])
MAX_TOOL_ATTEMPTS = int(os.getenv("EXECUTOR_MAX_ATTEMPTS", "3"))


app = FastAPI(title="Void Executor", version="0.1.0")
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s:%(name)s:%(message)s")


@app.on_event("startup")
async def _startup():
    try:
        from executor.utc.db import ensure_tables
        await ensure_tables()
        logging.info("[executor] UTC tables ensured")
    except Exception:
        logging.error(traceback.format_exc())

def within_workspace(path: str) -> str:
    full = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not full.startswith(os.path.abspath(WORKSPACE_DIR)):
        raise ValueError("path escapes workspace")
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
            try:
                exts = [str(x).lower() for x in val]
                vids = [x for x in exts if any(x.endswith(e) for e in (".mp4", ".mov", ".mkv", ".webm"))]
                auds = [x for x in exts if any(x.endswith(e) for e in (".wav", ".mp3", ".flac", ".aac"))]
                if vids:
                    out["videos_count"] = max(int(out.get("videos_count") or 0), len(vids))
                if auds:
                    out["audio_files_count"] = max(int(out.get("audio_files_count") or 0), len(auds))
            except Exception:
                pass
    return out


def _canonical_tool_result(x: Dict[str, Any] | Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if isinstance(x, dict) and "ok" in x and isinstance(x.get("result"), dict):
        data = x.get("result") or {}
    elif isinstance(x, dict):
        data = x
    ids = data.get("ids") if isinstance(data.get("ids"), dict) else {}
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    # Lift common meta-like fields from top-level bare results to prevent accidental drops
    lift_keys = (
        "url", "view_url", "poster_data_url", "preview_url",
        "mime", "duration_s", "preview_duration_s", "data_url",
    )
    for k in lift_keys:
        if k in data and k not in meta:
            v = data.get(k)
            if isinstance(v, (str, int, float, bool)) or v is None:
                meta[k] = v
    return {"ids": ids, "meta": meta}

@app.post("/execute_plan")
async def execute_plan(body: Dict[str, Any]):
    # Accept either {plan: {...}} or a raw plan object
    plan_obj = body.get("plan") if isinstance(body, dict) and isinstance(body.get("plan"), dict) else (body if isinstance(body, dict) else None)
    if not isinstance(plan_obj, dict):
        return JSONResponse(status_code=400, content={"error": "missing_or_invalid_plan"})

    raw_steps = plan_obj.get("plan") or []
    if not isinstance(raw_steps, list):
        return JSONResponse(status_code=422, content={"error": "invalid_plan_type"})

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
        # Attach trace metadata for orchestrator tracing
        rid = plan_obj.get("request_id") if isinstance(plan_obj.get("request_id"), str) else None
        if rid:
            inputs.setdefault("cid", rid)
            inputs.setdefault("trace_id", rid)
        sid = (step.get("id") or "").strip()
        if sid:
            inputs.setdefault("step_id", sid)
        return inputs

    pending = set(norm_steps.keys())
    trace_id = plan_obj.get("request_id") if isinstance(plan_obj.get("request_id"), str) else None
    logging.info(f"[executor] plan_start trace_id={trace_id} steps={len(pending)}")
    try:
        _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
            "t": int(time.time()*1000), "event": "exec_plan_start", "trace_id": trace_id, "steps": len(pending)
        }, expected={})
    except Exception:
        logging.error(traceback.format_exc())
    while pending:
        satisfied = set(produced.keys())
        runnable = [sid for sid in pending if deps[sid].issubset(satisfied)]
        if not runnable:
            return JSONResponse(status_code=422, content={"error": "deadlock_or_missing", "pending": sorted(list(pending))})
        batch_tools = [{"step_id": sid, "tool": norm_steps[sid].get("tool")} for sid in runnable]
        logging.info(f"[executor] batch_start trace_id={trace_id} runnable={batch_tools}")
        try:
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                "t": int(time.time()*1000), "event": "exec_batch_start", "trace_id": trace_id, "items": batch_tools
            }, expected={})
        except Exception:
            logging.error(traceback.format_exc())
        for sid in runnable:
            step = norm_steps[sid]
            tool_name = (step.get("tool") or "").strip()
            if not tool_name:
                return JSONResponse(status_code=422, content={"error": "missing_tool", "step": sid})
            args = _merge_inputs(step)
            # UTC-enabled tool run (builder→validate→autofix→retarget)
            t0 = time.time()
            # Distilled executor step start
            try:
                _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                    "t": int(time.time()*1000), "event": "exec_step_start", "tool": tool_name,
                    "trace_id": plan_obj.get("request_id"), "cid": plan_obj.get("request_id"), "step_id": sid
                }, expected={})
            except Exception:
                logging.error(traceback.format_exc())
            ok = False
            err_detail = None
            err_tb = None
            # Use UTC runner
            try:
                from executor.utc.utc_runner import utc_run_tool  # lazy import
                res = await utc_run_tool(trace_id, sid, tool_name, args)
            except Exception as e:
                err_detail = str(e)
                err_tb = traceback.format_exc()
                logging.error(err_tb)
                res = None
            else:
                if isinstance(res, dict) and bool(res.get("ok")) is True and res.get("result") is not None:
                    produced[sid] = res.get("result")
                    ok = True
                elif isinstance(res, dict) and bool(res.get("ok")) is False:
                    try:
                        err_obj = res.get("error") or {}
                        code = (err_obj.get("code") or "tool_error")
                        msg = (err_obj.get("message") or "")
                        tb = (err_obj.get("traceback") or "")
                        err_detail = f"{code}:{msg}"
                        if tb:
                            logging.error(str(tb))
                    except Exception:
                        err_detail = "tool_error"
                else:
                    produced[sid] = (res or {})
                    ok = True
            # Append end event to orchestrator tool logs for robust per-step trace
            try:
                _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                    "t": int(time.time()*1000),
                    "event": "end",
                    "tool": tool_name,
                    "ok": bool(ok),
                    "duration_ms": int((time.time() - t0) * 1000.0),
                    "trace_id": plan_obj.get("request_id"),
                    "cid": plan_obj.get("request_id"),
                    "step_id": sid,
                    "error": (None if ok else (err_detail or "tool_error")),
                    "traceback": (None if ok else err_tb),
                    "summary": (_distill_summary(produced.get(sid)) if ok else None),
                }, expected={"ok": bool, "error": str})
            except Exception:
                logging.error(traceback.format_exc())
            # Distilled executor step finish (separate event)
            try:
                _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                    "t": int(time.time()*1000), "event": "exec_step_finish", "tool": tool_name,
                    "trace_id": plan_obj.get("request_id"), "cid": plan_obj.get("request_id"), "step_id": sid,
                    "ok": bool(ok)
                }, expected={})
            except Exception:
                logging.error(traceback.format_exc())
            if err_detail is not None and not ok:
                return JSONResponse(status_code=422, content={"error": "tool_error", "step": sid, "detail": err_detail, "traceback": err_tb, "attempts": int(max(1, MAX_TOOL_ATTEMPTS))})
        for sid in runnable:
            pending.remove(sid)
        logging.info(f"[executor] batch_finish trace_id={trace_id} runnable={batch_tools}")
        try:
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                "t": int(time.time()*1000), "event": "exec_batch_finish", "trace_id": trace_id, "items": batch_tools
            }, expected={})
        except Exception:
            logging.error(traceback.format_exc())
    logging.info(f"[executor] plan_finish trace_id={trace_id} produced_keys={sorted(list(produced.keys()))}")
    try:
        _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
            "t": int(time.time()*1000), "event": "exec_plan_finish", "trace_id": trace_id, "produced_keys": sorted(list(produced.keys()))
        }, expected={})
    except Exception:
        logging.error(traceback.format_exc())
    return {"produced": produced}


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
        step = {
            "id": sid,
            "tool": (s.get("tool") or "").strip(),
            "inputs": dict(s.get("inputs") or {}),
            "needs": list(s.get("needs") or []),
        }
        norm_steps[sid] = step

    dependencies: Dict[str, set[str]] = {sid: set((norm_steps[sid].get("needs") or [])) for sid in norm_steps}
    produced: Dict[str, Any] = {}

    def merge_inputs(step: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(step.get("inputs") or {})
        for need in (step.get("needs") or []):
            if need in produced and isinstance(produced[need], dict):
                merged.update(produced[need])
        if request_id:
            merged.setdefault("cid", request_id)
            merged.setdefault("trace_id", request_id)
        sid_local = (step.get("id") or "").strip()
        if sid_local:
            merged.setdefault("step_id", sid_local)
        return merged

    pending = set(norm_steps.keys())
    logging.info(f"[executor] steps_start trace_id={trace_id} steps={len(pending)}")
    try:
        _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
            "t": int(time.time()*1000), "event": "exec_plan_start", "trace_id": request_id, "steps": len(pending)
        }, expected={})
    except Exception:
        logging.error(traceback.format_exc())

    while pending:
        satisfied = set(produced.keys())
        runnable = [sid for sid in pending if dependencies[sid].issubset(satisfied)]
        if not runnable:
            raise ValueError("deadlock_or_missing")
        batch_tools = [{"step_id": sid, "tool": norm_steps[sid].get("tool")} for sid in runnable]
        logging.info(f"[executor] batch_start trace_id={trace_id} runnable={batch_tools}")
        try:
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                "t": int(time.time()*1000), "event": "exec_batch_start", "trace_id": request_id, "items": batch_tools
            }, expected={})
        except Exception:
            logging.error(traceback.format_exc())
        for sid in runnable:
            step = norm_steps[sid]
            tool_name = (step.get("tool") or "").strip()
            if not tool_name:
                raise ValueError(f"missing_tool:{sid}")
            args = merge_inputs(step)
            t0 = time.time()
            try:
                _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                    "t": int(time.time()*1000), "event": "exec_step_start", "tool": tool_name,
                    "trace_id": request_id, "cid": request_id, "step_id": sid
                }, expected={})
            except Exception:
                logging.error(traceback.format_exc())
            try:
                from executor.utc.utc_runner import utc_run_tool  # lazy import
                res = await utc_run_tool(trace_id, sid, tool_name, args)
            except Exception as e:
                err_detail = str(e)
                err_tb = traceback.format_exc()
                logging.error(err_tb)
                try:
                    _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                        "t": int(time.time()*1000),
                        "event": "end",
                        "tool": tool_name,
                        "ok": False,
                        "duration_ms": int((time.time() - t0) * 1000.0),
                        "trace_id": request_id,
                        "cid": request_id,
                        "step_id": sid,
                        "error": err_detail,
                        "traceback": err_tb,
                        "summary": None,
                    }, expected={"ok": bool, "error": str})
                except Exception:
                    logging.error(traceback.format_exc())
                # Return structured failure instead of propagating 500
                res = {"schema_version": 1, "ok": False, "error": {"code": "executor_exception", "message": err_detail}}
            ok = False
            if isinstance(res, dict) and bool(res.get("ok")) is True and res.get("result") is not None:
                produced[sid] = {"name": tool_name, "result": _canonical_tool_result(res)}
                ok = True
            elif isinstance(res, dict) and bool(res.get("ok")) is False:
                try:
                    err_obj = res.get("error") or {}
                    code = (err_obj.get("code") or "tool_error")
                    msg = (err_obj.get("message") or "")
                    tb = (err_obj.get("traceback") or "")
                    err_detail = f"{code}:{msg}"
                    if tb:
                        logging.error(str(tb))
                except Exception:
                    err_detail = "tool_error"
                # Record structured failure result for this step
                produced[sid] = {"name": tool_name, "result": {"ids": {}, "meta": {"error": err_detail}}}
            else:
                produced[sid] = {"name": tool_name, "result": _canonical_tool_result(res or {})}
                ok = True
            try:
                _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                    "t": int(time.time()*1000),
                    "event": "end",
                    "tool": tool_name,
                    "ok": bool(ok),
                    "duration_ms": int((time.time() - t0) * 1000.0),
                    "trace_id": request_id,
                    "cid": request_id,
                    "step_id": sid,
                    "error": (None if ok else "tool_error"),
                    "traceback": None,
                    "summary": (_distill_summary(produced.get(sid)) if ok else None),
                }, expected={"ok": bool, "error": str})
            except Exception:
                logging.error(traceback.format_exc())
            try:
                _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                    "t": int(time.time()*1000), "event": "exec_step_finish", "tool": tool_name,
                    "trace_id": request_id, "cid": request_id, "step_id": sid,
                    "ok": bool(ok)
                }, expected={})
            except Exception:
                logging.error(traceback.format_exc())
        for sid in runnable:
            pending.remove(sid)
        logging.info(f"[executor] batch_finish trace_id={trace_id} runnable={batch_tools}")
        try:
            _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
                "t": int(time.time()*1000), "event": "exec_batch_finish", "trace_id": request_id, "items": batch_tools
            }, expected={})
        except Exception:
            logging.error(traceback.format_exc())
    logging.info(f"[executor] steps_finish trace_id={trace_id} produced_keys={sorted(list(produced.keys()))}")
    try:
        _post_json(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", {
            "t": int(time.time()*1000), "event": "exec_plan_finish", "trace_id": request_id, "produced_keys": sorted(list(produced.keys()))
        }, expected={})
    except Exception:
        logging.error(traceback.format_exc())
    return produced


@app.post("/execute")
async def execute_http(body: Dict[str, Any]):
    rid = str(body.get("request_id") or uuid.uuid4().hex)
    tid = str(body.get("trace_id") or uuid.uuid4().hex)
    steps = body.get("steps") if isinstance(body.get("steps"), list) else None

    if not steps:
        return JSONResponse(
            {"schema_version": 1, "request_id": rid, "ok": False,
             "error": {"code": "invalid_plan", "message": "steps required", "details": {}}},
            status_code=422
        )

    try:
        produced = await run_steps(tid, rid, steps)
        return JSONResponse(
            {"schema_version": 1, "request_id": rid, "ok": True, "result": {"produced": produced}},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            {"schema_version": 1, "request_id": rid, "ok": False, "error": {"code": "executor_exception", "message": str(e)}, "result": {"produced": []}},
            status_code=200
        )

