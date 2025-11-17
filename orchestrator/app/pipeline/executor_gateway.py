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
		return [{
			"name": "executor",
			"error": {
				"code": "executor_base_url_missing",
				"message": "executor_base_url is not configured",
				"stack": "".join(traceback.format_stack()),
			},
		}]
	async with _hx.AsyncClient(timeout=None, trust_env=False) as client:
		r = await client.post(base + "/execute", json=payload)
		parser = JSONParser()
		env = parser.parse(r.text or "{}", {"ok": bool, "result": dict, "error": dict})
		if not isinstance(env, dict):
			env = {
				"ok": False,
				"error": {
					"code": "executor_bad_json",
					"message": r.text or "",
					"stack": "".join(traceback.format_stack()),
				},
				"result": {"produced": {}},
			}
	results: List[Dict[str, Any]] = []
	if isinstance(env, dict) and env.get("ok") and isinstance((env.get("result") or {}).get("produced"), dict):
		for _, step in (env.get("result") or {}).get("produced", {}).items():
			if isinstance(step, dict):
				res = step.get("result") if isinstance(step.get("result"), dict) else {}
				results.append({"name": (res.get("name") or "tool") if isinstance(res, dict) else "tool", "result": res})
		return results
	err = (env or {}).get("error") or (env.get("result") or {}).get("error") or {}
	# Surface full structured error; never truncate to a generic 'executor_failed'.
	return [{
		"name": "executor",
		"error": {
			"code": err.get("code") or "executor_error",
			"message": err.get("message") or "executor_error",
			"stack": err.get("stack") or "".join(traceback.format_stack()),
		},
	}]


