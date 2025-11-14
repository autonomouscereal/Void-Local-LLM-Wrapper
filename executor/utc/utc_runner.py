import asyncio
import json
import time
import logging
import traceback
from typing import Any, Dict, Optional, List

import os
import random
from urllib.error import HTTPError
from .schema_fetcher import fetch as fetch_schema
from .validator import check as validate_args
from .universal_adapter import repair as repair_args
from .patch_store import preapply as preapply_patch, persist_success as persist_patch
from .db import get_pg_pool



def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            j = json.loads(body)
        except Exception:
            j = {}
        # Flatten FastAPI {"detail": {...}} to top-level envelope
        if isinstance(j, dict) and "detail" in j and isinstance(j["detail"], dict):
            j = j["detail"]
        if not isinstance(j, dict):
            j = {}
        j.setdefault("schema_version", 1)
        j.setdefault("ok", False)
        # Provide a code if missing
        if "code" not in j:
            j["code"] = "http_error"
        j["_http_status"] = e.code
        j["status"] = e.code
        return j
    except Exception:
        return {"schema_version": 1, "ok": False, "error": {"code": "network_error", "message": "request_failed"}}


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

    # Single global fixer pass (schema-driven), then validate and run
    v = validate_args(name, attempt_args, schema)
    # Use the universal adapter once to normalize based on schema/errors
    decision = repair_args(attempt_args, v.get("errors") or [], schema)
    attempt_args = decision.get("fixed_args") or attempt_args
    last_ops = decision.get("ops") or []
    await _emit_review_event("edit.plan", trace_id, step_id, notes=last_ops)
    await _emit_review_event("fixer.summary", trace_id, step_id, notes={"fixer_applied": bool(last_ops), "ops_count": len(last_ops)})

    # Validate once after fixer so required args are present (advisory-only)
    base = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
    venv = _post(base.rstrip("/") + "/tool.validate", {"name": name, "args": attempt_args})
    # Do not gate on validator outcome; always execute the tool once.
    attempt_args.setdefault("autofix_422", True)
    res = _post(base.rstrip("/") + "/tool.run", {"name": name, "args": attempt_args, "stream": False})
    if res.get("ok"):
        pool = await get_pg_pool()
        if pool is not None:
            try:
                async with pool.acquire() as c:
                    await c.execute("insert into tool_call_telemetry(trace_id,step_id,tool_name,version,builder_ok,repair_rounds,retargeted,status_code,error_json,final_args) values ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb)",
                                    trace_id, step_id, name, version, True, 1, False, 200, json.dumps(None), json.dumps(attempt_args))
            except Exception:
                pass
        return res

    # No retarget per canonical route set

    # final failure telemetry
    pool = await get_pg_pool()
    if pool is not None:
        try:
            async with pool.acquire() as c:
                await c.execute("insert into tool_call_telemetry(trace_id,step_id,tool_name,version,builder_ok,repair_rounds,retargeted,status_code,error_json,final_args) values ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb)",
                                trace_id, step_id, name, version, True, 1, False, int(res.get('status') or res.get('_http_status') or 422), json.dumps((res or {}).get("error")), json.dumps(attempt_args))
        except Exception:
            pass
    return res


