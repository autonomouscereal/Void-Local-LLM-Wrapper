import asyncio
import json
import time
from typing import Any, Dict, Optional, List

from ..app.main import ORCHESTRATOR_BASE_URL
from .schema_fetcher import fetch as fetch_schema
from .validator import check as validate_args
from .universal_adapter import repair as repair_args
from .patch_store import preapply as preapply_patch, persist_success as persist_patch
from .db import get_pg_pool
from .retarget import find_candidate


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except Exception:
        return {}


async def _emit_review_event(event: str, trace_id: Optional[str], step_id: Optional[str], notes: Any = None):
    try:
        payload = {
            "t": int(time.time() * 1000),
            "event": event,
            "trace_id": trace_id,
            "step_id": step_id,
            "notes": notes,
        }
        _post(ORCHESTRATOR_BASE_URL.rstrip("/") + "/logs/tools.append", payload)
    except Exception:
        pass


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
    # Try up to 3 rounds before retarget
    for round_idx in range(0, 3):
        v = validate_args(name, attempt_args, schema)
        if v.get("ok"):
            # call tool
            res = _post(ORCHESTRATOR_BASE_URL.rstrip("/") + "/tool.run", {"name": name, "args": attempt_args, "stream": False})
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

    # Retarget once
    alt = find_candidate(name)
    if alt:
        alt_name = alt.get("name")
        await _emit_review_event("review.decision", trace_id, step_id, notes=[f"retarget {name}→{alt_name}"])
        sch2 = await fetch_schema(alt_name)
        schema2 = sch2.get("schema") or {}
        v2 = validate_args(alt_name, attempt_args, schema2)
        if not v2.get("ok"):
            fix2 = repair_args(attempt_args, v2.get("errors") or [], schema2)
            attempt_args = fix2.get("fixed_args") or attempt_args
        res2 = _post(ORCHESTRATOR_BASE_URL.rstrip("/") + "/tool.run", {"name": alt_name, "args": attempt_args, "stream": False})
        # record telemetry
        pool = await get_pg_pool()
        if pool is not None:
            try:
                async with pool.acquire() as c:
                    await c.execute("insert into tool_call_telemetry(trace_id,step_id,tool_name,version,builder_ok,repair_rounds,retargeted,status_code,error_json,final_args) values ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb)",
                                    trace_id, step_id, alt_name, sch2.get("version") or "1", True, 1, True, (200 if res2.get('ok') else 422), json.dumps(res2.get("error") or {}), json.dumps(attempt_args))
            except Exception:
                pass
        return res2

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


