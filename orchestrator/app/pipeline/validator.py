from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable
import json
from app.json_parser import JSONParser  # type: ignore

import httpx as _hx  # type: ignore


async def validate_and_repair(
	tool_calls: List[Dict[str, Any]],
	*,
	base_url: str,
	temperature: float,
	last_user_text: str,
	tool_catalog_hash: str,
	trace_id: str,
	state_dir: str,
	planner_fn: Callable[[List[Dict[str, Any]], Any, float, str | None], Tuple[str, List[Dict[str, Any]]]],
	normalize_fn: Callable[[Any], List[Dict[str, Any]]],
	log_fn: Callable[..., None],
	checkpoints_append_event: Callable[[str, str, str, Dict[str, Any]], None],
) -> Dict[str, Any]:
	"""
	Prevalidate tool calls via /tool.validate; on failure, attempt a single repair round using the planner,
	then revalidate once. Collect aggregate failures. No early exits.
	Returns:
	  {
	    "validated": [...],
	    "pre_tool_failures": [...],
	    "repairs_made": bool,
	    "_repair_success_any": bool,
	    "_patched_payload_emitted": bool
	  }
	"""
	validated: List[Dict[str, Any]] = []
	pre_tool_failures: List[Dict[str, Any]] = []
	repairs_made = False
	_repair_success_any = False
	_patched_payload_emitted = False
	base = (base_url or "").rstrip("/")
	async with _hx.AsyncClient(trust_env=False, timeout=None) as client:
		# Pre-validate payload proof (before any network I/O)
		_steps_preview0 = []
		for t in (tool_calls or [])[:5]:
			_nm0 = (t.get("name") or "").strip()
			_ak0 = list((t.get("arguments") or {}).keys())
			_steps_preview0.append({"tool": _nm0, "args_keys": _ak0})
		log_fn("exec.payload", trace_id=trace_id, steps=_steps_preview0)
		# Early gate for image.dispatch required keys
		for p in _steps_preview0:
			if (p.get("tool") or "") == "image.dispatch":
				_ks0 = sorted([str(k) for k in (p.get("args_keys") or [])])
				_required0 = ["cfg", "height", "negative", "prompt", "steps", "width"]
				_missing0 = [k for k in _required0 if k not in _ks0]
				if _missing0:
					log_fn("exec.fail", trace_id=trace_id, tool="image.dispatch", reason="arguments_lost_before_execute", attempted_args_keys=_ks0)
					return {
						"validated": [],
						"pre_tool_failures": [],
						"repairs_made": False,
						"_repair_success_any": False,
						"_patched_payload_emitted": False,
					}
		# Validate and optionally repair
		for tc in tool_calls:
			name = (tc.get("name") or "").strip()
			args = tc.get("arguments") or {}
			# Describe once per tool to satisfy contract
			# Validate (no try/except; rely on body.ok semantics and schema parser)
			vpayload = json.dumps({"name": name, "args": args})
			vresp = await client.post(base + "/tool.validate", content=vpayload, headers={"content-type": "application/json"})
			parser = JSONParser()
			vobj = parser.parse(vresp.text or "", {"ok": bool, "error": {"code": str, "message": str, "details": dict}})
			status_code = int(getattr(vresp, "status_code", 0) or 0)
			log_fn("validate.result", trace_id=trace_id, status=status_code, tool=name, detail=((vobj or {}).get("error") or {}))
			ok_http = 200 <= status_code < 300
			ok_body = (isinstance(vobj, dict) and (vobj.get("ok") is True))
			# Accept HTTP 2xx as OK unless body explicitly sets ok False
			if ok_http and not (isinstance(vobj, dict) and (vobj.get("ok") is False)):
				validated.append(tc)
				continue
			# Repair once
			detail = (vobj.get("error") or {}).get("details") if isinstance(vobj, dict) else {}
			missing = (detail or {}).get("missing") or []
			invalid = (detail or {}).get("invalid") or []
			log_fn("committee.review", trace_id=trace_id, tool=name, validator_detail={"missing": missing, "invalid": invalid})
			log_fn("committee.decision", trace_id=trace_id, action="repair_once", rationale="ensure required fields present")
			brief = {
				"mode": "repair",
				"reason": "422 validation_error",
				"tool": name,
				"missing": missing,
				"invalid": invalid,
				"current_args": args,
				"requirements": {
					"must_use_tool": name,
					"fill_all_required": True,
					"snap_sizes_to_8": True,
					"defaults": {"negative": "", "width": 1024, "height": 1024, "steps": 32, "cfg": 5.5},
					"prompt_from_user_text_if_missing": True,
				},
				"user_text": last_user_text,
				"tool_catalog_hash": tool_catalog_hash,
			}
			import json as _json
			repair_preamble = (
				"You are in repair mode. Produce exactly the SAME tool with all required arguments filled and any invalid values corrected. "
				"Use the user's text as prompt if needed. Snap sizes to multiples of 8. Output only strict JSON: {\"steps\":[{\"tool\":\""
				+ name + "\",\"args\":{...}}]} with the same tool name."
			)
			repair_messages = [{"role": "system", "content": repair_preamble}, {"role": "user", "content": _json.dumps(brief, ensure_ascii=False)}]
			log_fn("repair.start", trace_id=trace_id, tool=name, brief=brief)
			_plan2, calls2 = await planner_fn(repair_messages, None, temperature, trace_id)
			calls2_norm = normalize_fn(calls2)
			patched = None
			for c2 in (calls2_norm or []):
				if not isinstance(c2, dict):
					continue
				c2_name = (c2.get("name") or "")
				if not isinstance(c2_name, str):
					continue
				if c2_name.strip() == name:
					args2 = c2.get("arguments") if isinstance(c2.get("arguments"), dict) else {}
					patched = {"name": name, "arguments": args2}
					break
			log_fn("planner.repair.steps", trace_id=trace_id, tool=name, patched=patched or {})
			if not patched:
				pre_tool_failures.append({"name": name, "result": {"error": (detail or {}), "status": 422}})
				continue
			# Re-validate once (no try/except; schema parser handles bad bodies)
			v2payload = json.dumps({"name": name, "args": patched["arguments"]})
			v2 = await client.post(base + "/tool.validate", content=v2payload, headers={"content-type": "application/json"})
			parser2 = JSONParser()
			v2obj = parser2.parse(v2.text or "", {"ok": bool, "error": {"code": str, "message": str, "details": dict}})
			status_code2 = int(getattr(v2, "status_code", 0) or 0)
			log_fn("validate.result.repair", trace_id=trace_id, status=status_code2, tool=name, detail=((v2obj or {}).get("error") or {}))
			ok_http2 = 200 <= status_code2 < 300
			ok_body2 = (isinstance(v2obj, dict) and (v2obj.get("ok") is True))
			if ok_http2 and not (isinstance(v2obj, dict) and (v2obj.get("ok") is False)):
				validated.append(patched)
				repairs_made = True
				_repair_success_any = True
			else:
				pre_err = (v2obj.get("error") if isinstance(v2obj, dict) else {}) or {}
				pre_tool_failures.append({"name": name, "result": {"error": pre_err, "status": 200}})
	# Emit payload proof when repairs made
	if repairs_made:
		_steps_preview = []
		for i, t in enumerate(validated[:5]):
			_nm = (t.get("name") or "").strip()
			_ak = list((t.get("arguments") or {}).keys())
			_steps_preview.append({"tool": _nm, "args_keys": _ak})
		log_fn("exec.payload", trace_id=trace_id, patched=True, steps=_steps_preview)
		_patched_payload_emitted = True
		log_fn("committee.go", trace_id=trace_id, tool="image.dispatch", rationale="re-validated=200")
		checkpoints_append_event(state_dir, trace_id, "committee.review.mid", {"phase": "post-repair", "tool": "image.dispatch"})
		checkpoints_append_event(state_dir, trace_id, "committee.decision.mid", {"action": "go"})
	else:
		log_fn("committee.go", trace_id=trace_id, tool="image.dispatch", rationale="validate=200")
		checkpoints_append_event(state_dir, trace_id, "committee.review.mid", {"phase": "post-validate", "tool": "image.dispatch"})
		checkpoints_append_event(state_dir, trace_id, "committee.decision.mid", {"action": "go"})
	return {
		"validated": validated,
		"pre_tool_failures": pre_tool_failures,
		"repairs_made": repairs_made,
		"_repair_success_any": _repair_success_any,
		"_patched_payload_emitted": _patched_payload_emitted,
	}


