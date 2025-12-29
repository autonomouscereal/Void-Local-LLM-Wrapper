import time
import logging
import traceback
from typing import Any, Dict, Optional, List

import os
import random
import json

import requests
from void_json import JSONParser
from void_envelopes import _build_error_envelope
from .schema_fetcher import fetch as fetch_schema
from .universal_adapter import repair as repair_args
from .patch_store import preapply as preapply_patch, persist_success as persist_patch
from .db import get_pg_pool


def _post(url: str, payload: Dict[str, Any]):
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


async def _emit_review_event(event: str, trace_id: Optional[str], step_id: Optional[str], notes: Any = None, tool_name: Optional[str] = None, version: Optional[str] = None):
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
                                    trace_id, step_id, tool_name or "unknown", version or "1", True, 0, False, 0, json.dumps({"post_logs_error": str(e)}), json.dumps({"notes": notes}))
            except Exception as e2:
                import logging, traceback
                logging.error(f"telemetry write failed: {e2!r}", exc_info=True)


RETRYABLE_STATUS = {429, 503, 524, 599}
RETRYABLE_ERRKINDS = {"network_reset", "connection_refused"}


def _is_transient(err: Dict[str, Any]):
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


def _stack_str():
    return "".join(traceback.format_stack())


async def utc_run_tool(trace_id: str, conversation_id: str, step_id: Optional[str], tool_name: str, args: Dict[str, Any]):
    sch = await fetch_schema(tool_name=tool_name)
    version = sch.get("version") or "1"
    schema = sch.get("schema") or {}
    schema_hash = sch.get("schema_hash") or ""

    # Primary builder (reuse existing args by default)
    built_args = dict(args)
    attempt_args = await preapply_patch(tool_name=tool_name, version=version, from_hash=schema_hash, args=built_args)
    last_ops: List[Dict[str, Any]] = []

    # WS narration start
    await _emit_review_event("review.start", trace_id, step_id, notes={"tool": tool_name}, tool_name=tool_name, version=version)

    # Single global fixer pass (schema-driven), then run (validator disabled)
    decision = repair_args(args=attempt_args, errors=[], schema=schema)
    attempt_args = decision.get("fixed_args") or attempt_args
    last_ops = decision.get("ops") or []
    await _emit_review_event("edit.plan", trace_id, step_id, notes=last_ops, tool_name=tool_name, version=version)
    await _emit_review_event("fixer.summary", trace_id, step_id, notes={"fixer_applied": bool(last_ops), "ops_count": len(last_ops)}, tool_name=tool_name, version=version)

    # Directly execute the tool once via /tool.run (no /tool.validate pre-check).
    base = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
    attempt_args.setdefault("autofix_422", True)
    # Always forward ids as provided by the caller (no fabrication, no fallbacks).
    # Tool routes still accept missing/empty ids; they must never crash.
    payload = {"tool_name": tool_name, "args": attempt_args, "stream": False, "trace_id": trace_id, "conversation_id": conversation_id}
    res = _post(base.rstrip("/") + "/tool.run", payload)

    # If orchestrator returned a non-dict, surface that as a structured error envelope.
    if not isinstance(res, dict):
        return _build_error_envelope(
            code="utc_non_dict_response",
            message=f"_post returned non-dict for tool {tool_name}",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=422,
            details={"raw": res, "stack": _stack_str()},
        )

    # Ensure we always attach a trace_id to the response envelope.
    if "trace_id" not in res or res.get("trace_id") is None:
        res["trace_id"] = trace_id
    if "conversation_id" not in res or res.get("conversation_id") is None:
        res["conversation_id"] = conversation_id

    # Guard against envelopes that fail to include an explicit ok flag. These
    # are treated as hard failures so executor logs always see a concrete code
    # and message instead of (code=None, msg=None, detail=None).
    if "ok" not in res:
        status = int(res.get("status") or res.get("_http_status") or 422)
        return _build_error_envelope(
            code="tool_missing_ok",
            message=f"orchestrator /tool.run did not return an 'ok' field for tool {tool_name}",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=status,
            details={"raw": res, "stack": _stack_str()},
        )

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
                        tool_name,
                        version,
                        True,
                        1,
                        False,
                        int(res.get("_http_status") or 200),
                        _json.dumps(None),
                        _json.dumps(attempt_args),
                    )
            except Exception as e:
                logging.error(f"tool_call_telemetry insert failed: {e!r}", exc_info=True)
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
    #
    # IMPORTANT: some tools return a structured stack/tool_stack object (dict/list),
    # while others return a plain string traceback. Preserve both without changing
    # types mid-flight.
    tool_stack: Any = None
    containers = [
        err_obj,
        (err_obj.get("details") if isinstance(err_obj, dict) and isinstance(err_obj.get("details"), dict) else None),
        res,
    ]
    for c in containers:
        if not isinstance(c, dict):
            continue
        for k in ("tool_stack", "traceback", "stack", "stacktrace"):
            v = c.get(k)
            if v is None:
                continue
            if isinstance(v, (dict, list)) and v:
                tool_stack = v
                break
            if isinstance(v, (str, bytes)) and v:
                # If this is actually JSON, keep it structured (tools sometimes return JSON strings).
                try:
                    parsed = JSONParser().parse(v, {})
                    if isinstance(parsed, (dict, list)) and parsed:
                        tool_stack = parsed
                    else:
                        tool_stack = v
                except Exception as ex:
                    logging.debug("utc_runner: JSONParser.parse failed for tool_stack (non-fatal) tool_name=%s trace_id=%s ex=%s", tool_name, trace_id, ex)
                    tool_stack = v
                break
        if tool_stack is not None:
            break
    # Always include a human-readable stack string as `stack`; preserve any structured tool stack
    # separately so downstream callers can safely do `.get(...)` on it.
    final_stack = tool_stack if isinstance(tool_stack, (str, bytes)) and tool_stack else _stack_str()
    return _build_error_envelope(
        code=str(err_obj.get("code") or "tool_error"),
        message=str(err_obj.get("message") or f"tool {tool_name} failed"),
        trace_id=trace_id,
        conversation_id=conversation_id,
        status=status,
        details={
            "raw": res,
            "tool_stack": (tool_stack if isinstance(tool_stack, (dict, list)) else None),
            "stack": final_stack,
        },
    )


