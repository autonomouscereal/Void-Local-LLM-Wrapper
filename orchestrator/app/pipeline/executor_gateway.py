from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx as _hx  # type: ignore
import uuid as _uuid
import traceback
from ..json_parser import JSONParser


async def execute(tool_calls: List[Dict[str, Any]], trace_id: Optional[str], executor_base_url: str) -> List[Dict[str, Any]]:
    """
    Execute tool calls via the external executor /execute endpoint.
    Returns a per-step list with either {"name":..., "result": {...}} or {"name":"executor","error":...}.
    Never raises; uses body.ok semantics from the executor envelope.
    """
    steps: List[Dict[str, Any]] = []
    for call in (tool_calls or [])[:5]:
        call_dict = call if isinstance(call, dict) else {}
        name_val = call_dict.get("name") if isinstance(call_dict.get("name"), str) else ""
        args_val = call_dict.get("arguments") if isinstance(call_dict.get("arguments"), dict) else {}
        steps.append({"tool": str(name_val or ""), "args": args_val})
    rid = str(trace_id or _uuid.uuid4().hex)
    payload = {"schema_version": 1, "request_id": rid, "trace_id": rid, "steps": steps}
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
    async with _hx.AsyncClient(timeout=None, trust_env=False) as client:
        r = await client.post(base + "/execute", json=payload)
        parser = JSONParser()
        raw_body = r.text or ""
        # The executor is expected to return a JSON envelope of the form
        # {"ok": bool, "result": {"produced": {...}}, "error": {...}}. When the
        # body is not valid JSON, or the shape drifts, we still want to surface
        # the raw response text and HTTP status instead of collapsing to a
        # generic executor_error.
        schema = {"ok": bool, "result": dict, "error": dict}
        env = parser.parse_superset(raw_body or "{}", schema)["coerced"]
        if not isinstance(env, dict):
            env = {
                "ok": False,
                "error": {
                    "code": "executor_bad_json",
                    "message": raw_body,
                    "status": int(r.status_code),
                    "stack": "".join(traceback.format_stack()),
                },
                "result": {"produced": {}},
            }
    results: List[Dict[str, Any]] = []
    if isinstance(env, dict) and env.get("ok") and isinstance((env.get("result") or {}).get("produced"), dict):
        produced = (env.get("result") or {}).get("produced", {}) or {}
        for _, step in produced.items():
            if not isinstance(step, dict):
                continue
            res = step.get("result") if isinstance(step.get("result"), dict) else {}
            tool_name = step.get("name") if isinstance(step.get("name"), str) else "tool"
            results.append({"name": tool_name, "result": res})
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
            },
        }
    ]
