from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging
import httpx as _hx  # type: ignore
import uuid as _uuid
import traceback
from ..json_parser import JSONParser

log = logging.getLogger(__name__)


async def execute(
    tool_calls: List[Dict[str, Any]],
    trace_id: Optional[str],
    executor_base_url: str,
    *,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Execute tool calls via the external executor /execute endpoint.
    Returns a per-step list with either {"name":..., "result": {...}} or {"name":"executor","error":...}.
    Never raises; uses body.ok semantics from the executor envelope.
    """
    steps: List[Dict[str, Any]] = []
    parser = JSONParser()
    for call in (tool_calls or []):
        call_dict = call if isinstance(call, dict) else {}
        name_val = call_dict.get("name") if isinstance(call_dict.get("name"), str) else ""
        raw_args = call_dict.get("arguments", call_dict.get("args"))
        # IMPORTANT: never silently drop tool arguments. The executor expects an
        # object for step.args, so when arguments isn't a dict we preserve it
        # under "_raw" rather than coercing to {}.
        args_val: Dict[str, Any]
        if isinstance(raw_args, dict):
            args_val = dict(raw_args)
        elif isinstance(raw_args, str):
            parsed = parser.parse(raw_args, {})
            args_val = dict(parsed) if isinstance(parsed, dict) else {"_raw": raw_args}
        elif raw_args is None:
            args_val = {}
        else:
            args_val = {"_raw": raw_args}
        steps.append({"tool": str(name_val or ""), "args": args_val})

    # request_id (rid) and trace_id are distinct identifiers; never reuse one as the other.
    rid = str(request_id or _uuid.uuid4().hex)
    tid = str(trace_id or _uuid.uuid4().hex)
    payload = {"schema_version": 1, "request_id": rid, "trace_id": tid, "steps": steps}
    try:
        # Never log full args payloads (may contain long prompts/base64). Log keys + types only.
        steps_preview = []
        for st in steps[:5]:
            a = st.get("args") if isinstance(st.get("args"), dict) else {}
            steps_preview.append(
                {
                    "tool": st.get("tool"),
                    "args_keys": sorted([str(k) for k in a.keys()])[:64],
                    "args_types": {str(k): type(v).__name__ for k, v in list(a.items())[:32]},
                }
            )
        log.info("executor_gateway.execute start trace_id=%s request_id=%s steps=%s preview=%s", tid, rid, len(steps), steps_preview)
    except Exception:
        log.debug("executor_gateway.execute: failed to emit start log preview", exc_info=True)
    base = (executor_base_url or "").rstrip("/")
    if not base:
        return [
            {
                "name": "executor",
                "error": {
                    "code": "executor_base_url_missing",
                    "message": "executor_base_url is not configured",
                    "stack": "".join(traceback.format_stack()),
                },
            }
        ]
    # IMPORTANT: this function must never raise. Tool execution failures should
    # be represented as per-step error objects so the caller can continue the run.
    try:
        async with _hx.AsyncClient(timeout=None, trust_env=False) as client:
            log.info("executor_gateway.execute POST %s/execute trace_id=%s request_id=%s", base, tid, rid)
            r = await client.post(base + "/execute", json=payload)
            raw_body = r.text or ""
    except _hx.TimeoutException as ex:  # type: ignore[attr-defined]
        return [
            {
                "name": "executor",
                "error": {
                    "code": "executor_timeout",
                    "message": str(ex),
                    "status": 0,
                    "stack": "".join(traceback.format_exc()),
                    "executor_base_url": base,
                },
            }
        ]
    except _hx.RequestError as ex:  # type: ignore[attr-defined]
        return [
            {
                "name": "executor",
                "error": {
                    "code": "executor_request_error",
                    "message": str(ex),
                    "status": 0,
                    "stack": "".join(traceback.format_exc()),
                    "executor_base_url": base,
                },
            }
        ]
    except Exception as ex:
        return [
            {
                "name": "executor",
                "error": {
                    "code": "executor_exception",
                    "message": str(ex),
                    "status": 0,
                    "stack": "".join(traceback.format_exc()),
                    "executor_base_url": base,
                },
            }
        ]

    # The executor is expected to return a JSON envelope of the form
    # {"ok": bool, "result": {"produced": {...}}, "error": {...}}. When the
    # body is not valid JSON, or the shape drifts, we still want to surface
    # the raw response text and HTTP status instead of collapsing to a
    # generic executor_error.
    try:
        schema = {"ok": bool, "result": dict, "error": dict}
        env = parser.parse(raw_body or "{}", schema)
    except Exception:
        env = {
            "ok": False,
            "error": {
                "code": "executor_bad_json",
                "message": raw_body,
                "status": int(getattr(r, "status_code", 0) or 0),
                "stack": "".join(traceback.format_exc()),
            },
            "result": {"produced": {}},
        }
    if not isinstance(env, dict):
        env = {
            "ok": False,
            "error": {
                "code": "executor_bad_json",
                "message": raw_body,
                "status": int(getattr(r, "status_code", 0) or 0),
                "stack": "".join(traceback.format_stack()),
            },
            "result": {"produced": {}},
        }
    try:
        ok_flag = bool(env.get("ok")) if isinstance(env, dict) else False
        produced_obj = (env.get("result") or {}).get("produced") if isinstance(env, dict) else None
        produced_keys = sorted(list(produced_obj.keys()))[:32] if isinstance(produced_obj, dict) else []
        log.info(
            "executor_gateway.execute response trace_id=%s request_id=%s http_status=%s ok=%s produced_keys=%s",
            tid,
            rid,
            int(r.status_code),
            ok_flag,
            produced_keys,
        )
    except Exception:
        log.debug("executor_gateway.execute: failed to emit response log summary", exc_info=True)
    results: List[Dict[str, Any]] = []
    if isinstance(env, dict) and env.get("ok") and isinstance((env.get("result") or {}).get("produced"), dict):
        produced = (env.get("result") or {}).get("produced", {}) or {}
        # Attach args back to results (executor does not echo args).
        #
        # IMPORTANT: Do NOT sort produced and then match by enumerate index.
        # produced is a dict keyed by step id (e.g. "s1", "s2", ...), and dict
        # iteration order may not match the input tool-calls order. Index-based
        # matching can therefore attach the wrong args to a result.
        #
        # Instead, build a step-id -> input-step map using the implicit "s{i}"
        # convention used by the executor for list-ordered steps.
        input_step_by_sid: Dict[str, Dict[str, Any]] = {}
        for i, st in enumerate(steps):
            if isinstance(st, dict):
                input_step_by_sid[f"s{i + 1}"] = st
        for sid, step in produced.items():
            if not isinstance(step, dict):
                continue
            sid_str = str(sid)
            input_step = input_step_by_sid.get(sid_str, {})
            args_out = input_step.get("args") if isinstance(input_step.get("args"), dict) else {}
            input_tool = input_step.get("tool") if isinstance(input_step.get("tool"), str) else None
            tool_name = step.get("name") if isinstance(step.get("name"), str) else (input_tool or "tool")
            res = step.get("result") if isinstance(step.get("result"), dict) else {}

            # Preserve all executor-provided fields (e.g. artifacts/meta/timing),
            # but ensure the canonical tool-result keys are present/normalized.
            out: Dict[str, Any] = dict(step)
            out["name"] = tool_name
            out["result"] = res
            out["args"] = args_out
            out["step_id"] = sid_str
            results.append(out)
        return results
    err = (env or {}).get("error") or (env.get("result") or {}).get("error") or {}
    # Surface full structured error; never truncate to a generic 'executor_failed'.
    # Include HTTP status and the raw executor body when available so callers
    # can see exactly what the executor returned.
    return [
        {
            "name": "executor",
            "error": {
                "code": err.get("code") or "executor_error",
                "message": err.get("message") or "executor_error",
                "status": err.get("status") or err.get("_http_status") or int(r.status_code),
                "stack": err.get("stack") or "".join(traceback.format_stack()),
                "env": env,
                "body": raw_body,
                "executor_base_url": base,
            },
        }
    ]
