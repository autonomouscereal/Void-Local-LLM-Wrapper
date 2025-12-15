import time
import logging
import traceback
from typing import Any, Dict, Optional, List

import os
import random
import json

import requests
from void_json import JSONParser
from .schema_fetcher import fetch as fetch_schema
from .universal_adapter import repair as repair_args
from .patch_store import preapply as preapply_patch, persist_success as persist_patch
from .db import get_pg_pool


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous HTTP POST helper for executor → orchestrator calls.

    Returns a dict representing the parsed JSON body when possible and
    injects an `_http_status` field with the HTTP status code. No exceptions
    are caught here; connection-level failures should surface directly.
    """
    r = requests.post(url, json=payload, timeout=None)
    text = r.text or ""
    parser = JSONParser()
    # Executor → orchestrator calls are expected to return an envelope-like
    # body with ok/result/error. Use a shallow schema so callers always see a
    # dict with those keys, and attach the HTTP status out-of-band.
    env_schema = {"ok": bool, "result": dict, "error": dict}
    try:
        body = parser.parse(text or "{}", env_schema)
    except Exception as e:
        # Parsing should virtually never fail for orchestrator envelopes. When
        # it does, surface the raw body and parse error so executor logs can
        # explain exactly what went wrong instead of only reporting _http_status.
        return {
            "_http_status": r.status_code,
            "raw": text[:4000],
            "parse_error": repr(e),
        }
    if isinstance(body, dict):
        body.setdefault("_http_status", r.status_code)
        return body
    # Non-dict envelopes are treated as hard failures; preserve the raw body so
    # callers (and logs) can inspect what the orchestrator actually returned.
    return {"_http_status": r.status_code, "raw": body}


async def _emit_review_event(event: str, trace_id: Optional[str], step_id: Optional[str], notes: Any = None):
    try:
        payload = {
            "t": int(time.time() * 1000),
            "event": event,
            "trace_id": trace_id,
            "step_id": step_id,
            "notes": notes,
        }
        base = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
        _post(base.rstrip("/") + "/logs/tools.append", payload)
    except Exception as e:
        # Log locally to telemetry if posting fails
        pool = await get_pg_pool()
        if pool is not None:
            try:
                async with pool.acquire() as c:
                    await c.execute("insert into tool_call_telemetry(trace_id,step_id,tool_name,version,builder_ok,repair_rounds,retargeted,status_code,error_json,final_args) values ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb)",
                                    trace_id, step_id, str(event), "utc", True, 0, False, 0, json.dumps({"post_logs_error": str(e)}), json.dumps({"notes": notes}))
            except Exception as e2:
                import logging, traceback
                logging.error("telemetry write failed: %s", e2, exc_info=True)


RETRYABLE_STATUS = {429, 503, 524, 599}
RETRYABLE_ERRKINDS = {"network_reset", "connection_refused"}


def _is_transient(err: Dict[str, Any]) -> bool:
    msg = ((err or {}).get("message") or "").lower()
    code = ((err or {}).get("code") or "").lower()
    # never retry on timeouts by policy
    if "timeout" in msg or "timed out" in msg or code == "timeout":
        return False
    if code in ("temporarily_unavailable",):
        return True
    if "network" in msg or "connection" in msg or "rate" in msg:
        return True
    return False


def _stack_str() -> str:
    return "".join(traceback.format_stack())


async def utc_run_tool(trace_id: Optional[str], step_id: Optional[str], name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    sch = await fetch_schema(name)
    version = sch.get("version") or "1"
    schema = sch.get("schema") or {}
    schema_hash = sch.get("schema_hash") or ""

    # Primary builder (reuse existing args by default)
    built_args = dict(args)
    attempt_args = await preapply_patch(name, version, schema_hash, built_args)
    last_ops: List[Dict[str, Any]] = []

    # WS narration start
    await _emit_review_event("review.start", trace_id, step_id, notes={"tool": name})

    # Single global fixer pass (schema-driven), then run (validator disabled)
    decision = repair_args(attempt_args, [], schema)
    attempt_args = decision.get("fixed_args") or attempt_args
    last_ops = decision.get("ops") or []
    await _emit_review_event("edit.plan", trace_id, step_id, notes=last_ops)
    await _emit_review_event("fixer.summary", trace_id, step_id, notes={"fixer_applied": bool(last_ops), "ops_count": len(last_ops)})

    # Directly execute the tool once via /tool.run (no /tool.validate pre-check).
    base = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
    attempt_args.setdefault("autofix_422", True)
    payload = {"name": name, "args": attempt_args, "stream": False, "trace_id": trace_id}
    res = _post(base.rstrip("/") + "/tool.run", payload)

    # If orchestrator returned a non-dict, surface that as a structured error envelope.
    if not isinstance(res, dict):
        return {
            "schema_version": 1,
            "trace_id": trace_id,
            "ok": False,
            "result": None,
            "error": {
                "code": "utc_non_dict_response",
                "message": f"_post returned non-dict for tool {name}",
                "status": 422,
                "raw": res,
                "stack": _stack_str(),
            },
        }

    # Ensure we always attach a trace_id to the response envelope.
    if "trace_id" not in res or res.get("trace_id") is None:
        res["trace_id"] = trace_id

    # Guard against envelopes that fail to include an explicit ok flag. These
    # are treated as hard failures so executor logs always see a concrete code
    # and message instead of (code=None, msg=None, detail=None).
    if "ok" not in res:
        status = int(res.get("status") or res.get("_http_status") or 422)
        return {
            "schema_version": 1,
            "trace_id": trace_id,
            "ok": False,
            "result": None,
            "error": {
                "code": "tool_missing_ok",
                "message": f"orchestrator /tool.run did not return an 'ok' field for tool {name}",
                "status": status,
                "raw": res,
                "stack": _stack_str(),
            },
        }

    # If orchestrator indicates success (ok != False), pass envelope through.
    if res.get("ok") is not False:
        pool = await get_pg_pool()
        if pool is not None:
            try:
                import json as _json
                async with pool.acquire() as c:
                    await c.execute(
                        "insert into tool_call_telemetry(trace_id,step_id,tool_name,version,builder_ok,repair_rounds,retargeted,status_code,error_json,final_args) "
                        "values ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb)",
                        trace_id,
                        step_id,
                        name,
                        version,
                        True,
                        1,
                        False,
                        int(res.get("_http_status") or 200),
                        _json.dumps(None),
                        _json.dumps(attempt_args),
                    )
            except Exception as e:
                logging.error("tool_call_telemetry insert failed: %s", e, exc_info=True)
        return res

    # Non-ok orchestrator response: wrap with a canonical error envelope.
    err_obj = res.get("error") if isinstance(res.get("error"), dict) else {}
    # Prefer the tool-layer status when present, fall back to transport status.
    status = int(
        (err_obj.get("status") if isinstance(err_obj, dict) and isinstance(err_obj.get("status"), int) else None)
        or res.get("status")
        or res.get("_http_status")
        or 422
    )
    # Preserve the ORIGINAL tool traceback/stack when available.
    tool_stack: str | bytes | None = None
    containers = [
        err_obj,
        (err_obj.get("details") if isinstance(err_obj, dict) and isinstance(err_obj.get("details"), dict) else None),
        res,
    ]
    for c in containers:
        if not isinstance(c, dict):
            continue
        for k in ("traceback", "stack", "stacktrace"):
            v = c.get(k)
            if isinstance(v, (str, bytes)) and v:
                tool_stack = v
                break
        if tool_stack is not None:
            break
    final_stack = tool_stack if isinstance(tool_stack, (str, bytes)) and tool_stack else _stack_str()
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "ok": False,
        "result": None,
        "error": {
            "code": err_obj.get("code") or "tool_error",
            "message": err_obj.get("message") or f"tool {name} failed",
            "status": status,
            "raw": res,
            "stack": final_stack,
        },
    }


