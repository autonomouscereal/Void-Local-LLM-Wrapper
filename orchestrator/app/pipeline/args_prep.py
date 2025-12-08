from __future__ import annotations

from typing import Any, Dict, List, Callable


SERVER_ONLY_KEYS = {"trace_id", "cid", "step_id", "film_id", "segment_id"}


def ensure_object_args(tool_calls: List[Dict[str, Any]], parse_json: Callable[[str, Any], Any]) -> List[Dict[str, Any]]:
	"""
	Ensure each tool call has dict arguments. If arguments is a JSON string, parse to dict.
	Accepts a parse_json callback to avoid importing the app.main helper.
	"""
	out: List[Dict[str, Any]] = []
	for tc in tool_calls or []:
		args_obj = (tc or {}).get("arguments")
		if isinstance(args_obj, dict) and isinstance(args_obj.get("_raw"), str):
			parsed = parse_json(args_obj.get("_raw"), {})
			tc = {**tc, "arguments": (parsed if isinstance(parsed, dict) else {})}
		elif isinstance(args_obj, str):
			parsed = parse_json(args_obj, {})
			tc = {**tc, "arguments": (parsed if isinstance(parsed, dict) else {})}
		out.append(tc)
	return out


def strip_server_ids(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""
	Remove server-reserved id fields from tool arguments so models can never
	control trace_id/cid/step_id or internal film/segment ids.
	"""
	out: List[Dict[str, Any]] = []
	for tc in tool_calls or []:
		if not isinstance(tc, dict):
			continue
		args = (tc.get("arguments") or {})
		if not isinstance(args, dict):
			args = {}
		for k in SERVER_ONLY_KEYS:
			if k in args:
				args.pop(k, None)
		out.append({**tc, "arguments": args})
	return out


def fill_min_defaults(tool_calls: List[Dict[str, Any]], last_user_text: str, log_fn: Callable[..., None] | None = None, trace_id: str | None = None) -> List[Dict[str, Any]]:
	"""
	Fill minimal defaults for image.dispatch without clobbering user-provided values.
	Optionally logs the compact args snapshot for planner.image.dispatch.args via log_fn.
	"""
	enriched: List[Dict[str, Any]] = []
	for tc in tool_calls or []:
		nm = (tc.get("name") or "").strip()
		args = (tc.get("arguments") or {})
		if not isinstance(args, dict):
			args = {"_raw": args}
		if nm == "image.dispatch":
			if args.get("negative") is None:
				args["negative"] = ""
			if args.get("width") is None:
				args["width"] = 1024
			if args.get("height") is None:
				args["height"] = 1024
			if args.get("steps") is None:
				args["steps"] = 32
			if args.get("cfg") is None:
				args["cfg"] = 5.5
			if (not isinstance(args.get("prompt"), str)) or not (args.get("prompt") or "").strip():
				if isinstance(last_user_text, str) and last_user_text.strip():
					args["prompt"] = last_user_text.strip()
			if callable(log_fn):
				try:
					log_fn("planner.image.dispatch.args", trace_id=trace_id, args={k: args.get(k) for k in ("prompt", "width", "height", "steps", "cfg", "negative")})
				except Exception:
					pass
		enriched.append({**tc, "arguments": args})
	return enriched


