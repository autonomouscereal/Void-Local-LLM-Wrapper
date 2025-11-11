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
    # Preflight validate to surface args issues early (no retries/timeouts here)
    base = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
    venv = _post(base.rstrip("/") + "/tool.validate", {"name": name, "args": attempt_args})
    if isinstance(venv, dict) and venv.get("ok") is False:
        return {"schema_version": 1, "ok": False, "code": "args_invalid", "result": venv.get("result")}

    # Try up to 3 rounds before giving up
    for round_idx in range(0, 3):
        v = validate_args(name, attempt_args, schema)
        if v.get("ok"):
            # call tool
            attempt_args.setdefault("autofix_422", True)
            res = _post(base.rstrip("/") + "/tool.run", {"name": name, "args": attempt_args, "stream": False})
            if res.get("ok"):
                if round_idx > 0 and last_ops:
                    await persist_patch(name, version, schema_hash, schema_hash, last_ops)
                await _emit_review_event("review.decision", trace_id, step_id, notes=(f"accepted after {round_idx} repairs" if round_idx else "accepted"))
                # telemetry
                pool = await get_pg_pool()
                if pool is not None:
                    try:
                        async with pool.acquire() as c:
                            await c.execute("insert into tool_call_telemetry(trace_id,step_id,tool_name,version,builder_ok,repair_rounds,retargeted,status_code,error_json,final_args) values ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb)",
                                            trace_id, step_id, name, version, True, round_idx, False, 200, json.dumps(None), json.dumps(attempt_args))
                    except Exception:
                        pass
                return res
            # If orchestrator returned a structured error (including 422), attempt a single auto-fix and retry
            code = (res.get("code")
                    or (res.get("error") or {}).get("code")
                    or (res.get("result") or {}).get("code"))
            if code in ("invalid_workflow", "workflow_invalid", "workflow_binding_missing", "args_invalid", "tool_failed") or res.get("_http_status") == 422:
                # Simple bounded auto-fix
                attempt_args.setdefault("autofix_422", True)
                # Snap dims to multiples of 8 within [64,2048]
                def _snap_dim(k):
                    if k in attempt_args and attempt_args[k] is not None:
                        try:
                            v = int(float(attempt_args[k]))
                            v = max(64, min(2048, v - (v % 8)))
                            attempt_args[k] = v
                        except Exception:
                            pass
                _snap_dim("width"); _snap_dim("height")
                # Coerce numeric types/ranges
                try:
                    attempt_args["steps"] = int(float(attempt_args.get("steps", 30)))
                except Exception:
                    attempt_args["steps"] = 30
                try:
                    attempt_args["cfg"] = float(attempt_args.get("cfg", 7.0))
                except Exception:
                    attempt_args["cfg"] = 7.0
                if not attempt_args.get("sampler"):
                    attempt_args["sampler"] = "euler"
                if not attempt_args.get("scheduler"):
                    attempt_args["scheduler"] = "normal"
                try:
                    int(attempt_args.get("seed", ""))
                except Exception:
                    attempt_args["seed"] = random.randint(1, 2**31 - 1)
                res_retry = _post(base.rstrip("/") + "/tool.run", {"name": name, "args": attempt_args, "stream": False})
                if res_retry.get("ok"):
                    return res_retry
            else:
                err = res.get("error") or {}
                if _is_transient(err):
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** round_idx)))
                    continue
                # hard fail -> break to repair
        # v not ok or hard fail -> repair
        if round_idx == 2:
            break
        decision = repair_args(attempt_args, v.get("errors") or [], schema)
        attempt_args = decision.get("fixed_args") or attempt_args
        last_ops = decision.get("ops") or []
        await _emit_review_event("edit.plan", trace_id, step_id, notes=last_ops)
        await _emit_review_event("review.decision", trace_id, step_id, notes=f"autofix round {round_idx+1}")

    # No retarget per canonical route set

    # final failure telemetry
    pool = await get_pg_pool()
    if pool is not None:
        try:
            async with pool.acquire() as c:
                await c.execute("insert into tool_call_telemetry(trace_id,step_id,tool_name,version,builder_ok,repair_rounds,retargeted,status_code,error_json,final_args) values ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb)",
                                trace_id, step_id, name, version, True, 2, False, 422, json.dumps({"code": "tool_error", "message": "Unable to satisfy tool schema after bounded repairs."}), json.dumps(attempt_args))
        except Exception:
            pass
    return {"ok": False, "error": {"code": "tool_error", "message": "Unable to satisfy tool schema after bounded repairs.", "details": {"name": name}}}


