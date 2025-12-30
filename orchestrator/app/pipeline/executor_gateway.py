from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging
import httpx as _hx  # type: ignore
import traceback
from ..json_parser import JSONParser

log = logging.getLogger(__name__)


async def execute(
    tool_calls: List[Dict[str, Any]],
    trace_id: str,
    conversation_id: str,
    executor_base_url: str,
) -> List[Dict[str, Any]]:
    """
    Execute tool calls via the external executor /execute endpoint.
    Returns a per-step list with either {"name":..., "result": {...}} or {"name":"executor","error":...}.
    Never raises; uses body.ok semantics from the executor envelope.
    """
    steps: List[Dict[str, Any]] = []
    parser = JSONParser()
    for tool_call in (tool_calls or []):
        tool_call_dict = tool_call if isinstance(tool_call, dict) else {}
        tool_name = tool_call_dict.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        tool_name = tool_name.strip()
        step_id = tool_call_dict.get("step_id") if isinstance(tool_call_dict.get("step_id"), str) else None
        raw_args = tool_call_dict.get("arguments", tool_call_dict.get("args"))
        # IMPORTANT: never silently drop tool arguments. The executor expects an
        # object for step.args, so when arguments isn't a dict we preserve it
        # under "_raw" rather than coercing to {}.
        args: Dict[str, Any]
        if isinstance(raw_args, dict):
            args = dict(raw_args)
        elif isinstance(raw_args, str):
            parsed = parser.parse(source=raw_args, expected_structure={})
            args = dict(parsed) if isinstance(parsed, dict) else {"_raw": raw_args}
        elif raw_args is None:
            args = {}
        else:
            args = {"_raw": raw_args}
        # Always propagate trace_id and conversation_id into args
        # CRITICAL: trace_id MUST be passed from chat_completions - always inject it
        if isinstance(args, dict):
            # Always inject trace_id if provided (even if empty string, it should be passed through)
            if trace_id and not args.get("trace_id"):
                args["trace_id"] = trace_id
            if conversation_id and not args.get("conversation_id"):
                args["conversation_id"] = conversation_id
        step_payload: Dict[str, Any] = {"tool": tool_name, "args": args}
        # Preserve planner-provided step_id so executor-produced keys match the plan.
        if isinstance(step_id, str) and step_id:
            step_payload["step_id"] = step_id
        steps.append(step_payload)

    payload = {"schema_version": 1, "trace_id": trace_id, "conversation_id": conversation_id, "steps": steps}
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
        log.info(
            "executor_gateway.execute start trace_id=%s conversation_id=%s steps=%d preview=%s",
            trace_id,
            conversation_id,
            int(len(steps)),
            steps_preview,
        )
    except Exception:
        log.debug("executor_gateway.execute: failed to emit start log preview", exc_info=True)
    base = (executor_base_url or "").rstrip("/")
    if not base:
        return [
            {
                "tool_name": "executor",
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
            log.info("executor_gateway.execute POST %s/execute trace_id=%s conversation_id=%s", base, trace_id, conversation_id)
            r = await client.post(base + "/execute", json=payload)
            raw_body = r.text or ""
    except _hx.TimeoutException as ex:  # type: ignore[attr-defined]
        return [
            {
                "tool_name": "executor",
                "error": {
                    "code": "executor_timeout",
                    "message": str(ex),
                    "status": 0,
                    "stack": traceback.format_exc(),
                    "executor_base_url": base,
                },
            }
        ]
    except _hx.RequestError as ex:  # type: ignore[attr-defined]
        return [
            {
                "tool_name": "executor",
                "error": {
                    "code": "executor_request_error",
                    "message": str(ex),
                    "status": 0,
                    "stack": traceback.format_exc(),
                    "executor_base_url": base,
                },
            }
        ]
    except Exception as ex:
        return [
            {
                "tool_name": "executor",
                "error": {
                    "code": "executor_exception",
                    "message": str(ex),
                    "status": 0,
                    "stack": traceback.format_exc(),
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
    except Exception as ex:
        log.warning("executor_gateway.execute: JSONParser.parse failed trace_id=%s conversation_id=%s ex=%s raw_body_prefix=%s", trace_id, conversation_id, ex, (raw_body or "")[:200], exc_info=True)
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
            "executor_gateway.execute response trace_id=%s http_status=%s ok=%s produced_keys=%s",
            trace_id,
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
        input_step_by_step_id: Dict[str, Dict[str, Any]] = {}
        for step_index, input_step in enumerate(steps):
            if not isinstance(input_step, dict):
                continue
            # When a step_id is supplied, the executor can use it as the produced key.
            key = input_step.get("step_id") if isinstance(input_step.get("step_id"), str) else None
            if not key:
                key = f"s{step_index + 1}"
            input_step_by_step_id[str(key)] = input_step
        for produced_step_id, produced_step in produced.items():
            if not isinstance(produced_step, dict):
                log.warning("executor_gateway.execute: skipping non-dict produced_step step_id=%s type=%s trace_id=%s", produced_step_id, type(produced_step).__name__, trace_id)
                continue
            step_id_str = str(produced_step_id)
            input_step = input_step_by_step_id.get(step_id_str, {})
            args_out = input_step.get("args") if isinstance(input_step.get("args"), dict) else {}
            tool_name = produced_step.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                log.warning("executor_gateway.execute: skipping produced_step with missing/invalid tool_name step_id=%s trace_id=%s", step_id_str, trace_id)
                continue
            tool_name = tool_name.strip()
            tool_payload = produced_step.get("result") if isinstance(produced_step.get("result"), dict) else {}

            # Preserve all executor-provided fields (e.g. artifacts/meta/timing),
            # but ensure the canonical tool-result keys are present/normalized.
            out: Dict[str, Any] = dict(produced_step)
            out["tool_name"] = tool_name
            out["result"] = tool_payload
            out["args"] = args_out
            out["step_id"] = step_id_str
            results.append(out)
        return results
    err = (env or {}).get("error") or (env.get("result") or {}).get("error") or {}
    # Surface full structured error; never truncate to a generic 'executor_failed'.
    # Include HTTP status and the raw executor body when available so callers
    # can see exactly what the executor returned.
    return [
        {
            "tool_name": "executor",
            "tool_call_name": "executor",  # legacy compatibility
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
