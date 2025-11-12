from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx as _hx  # type: ignore
import uuid as _uuid


async def execute(tool_calls: List[Dict[str, Any]], trace_id: Optional[str], executor_base_url: str) -> List[Dict[str, Any]]:
	"""
	Execute tool calls via the external executor /execute endpoint.
	Returns a per-step list with either {"name":..., "result": {...}} or {"name":"executor","error":...}.
	Never raises; uses body.ok semantics from the executor envelope.
	"""
	steps: List[Dict[str, Any]] = []
	for call in (tool_calls or [])[:5]:
		name = (call or {}).get("name")
		args = (call or {}).get("arguments") or {}
		if not isinstance(args, dict):
			args = {}
		steps.append({"tool": str(name or ""), "args": args})
	rid = str(trace_id or _uuid.uuid4().hex)
	payload = {"schema_version": 1, "request_id": rid, "trace_id": rid, "steps": steps}
	base = (executor_base_url or "").rstrip("/")
	if not base:
		return [{"name": "executor", "error": "executor_base_url_missing"}]
	async with _hx.AsyncClient(timeout=None, trust_env=False) as client:
		try:
			r = await client.post(base + "/execute", json=payload)
			try:
				env = r.json()
			except Exception:
				env = {"ok": False, "error": {"code": "executor_bad_json", "message": r.text}, "result": {"produced": {}}}
		except Exception as ex:
			env = {"ok": False, "error": {"code": "executor_connect_error", "message": str(ex)}, "result": {"produced": {}}}
	results: List[Dict[str, Any]] = []
	if isinstance(env, dict) and env.get("ok") and isinstance((env.get("result") or {}).get("produced"), dict):
		for _, step in (env.get("result") or {}).get("produced", {}).items():
			if isinstance(step, dict):
				res = step.get("result") if isinstance(step.get("result"), dict) else {}
				results.append({"name": (res.get("name") or "tool") if isinstance(res, dict) else "tool", "result": res})
		return results
	err = (env or {}).get("error") or (env.get("result") or {}).get("error") or {}
	return [{"name": "executor", "error": (err.get("message") or "executor_failed")}]


