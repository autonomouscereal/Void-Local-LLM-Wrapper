from __future__ import annotations

import logging
from typing import Any, Dict, List, Callable, Optional

from ..json_parser import JSONParser


def ensure_object_args(
    tool_calls: List[Dict[str, Any]],
    parse_json: Optional[Callable[[str, Any], Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Ensure each tool call has dict arguments. If arguments is a JSON string, parse to dict.
    If parse_json is not provided, uses JSONParser().parse.
    """
    parser = JSONParser()
    parse_fn = parse_json if callable(parse_json) else (lambda raw, schema: parser.parse(raw or "", schema))
    out: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        args_obj = (tc or {}).get("arguments")
        if isinstance(args_obj, dict) and isinstance(args_obj.get("_raw"), str):
            raw = args_obj.get("_raw")
            parsed = parse_fn(raw, dict)
            # Never drop args: if parsing fails, preserve the original string.
            tc = {**tc, "arguments": (dict(parsed) if isinstance(parsed, dict) else {"_raw": raw})}
        elif isinstance(args_obj, str):
            parsed = parse_fn(args_obj, dict)
            # Never drop args: if parsing fails, preserve the original string.
            tc = {**tc, "arguments": (dict(parsed) if isinstance(parsed, dict) else {"_raw": args_obj})}
        elif args_obj is None:
            tc = {**tc, "arguments": {}}
        elif not isinstance(args_obj, dict):
            # Preserve any non-object args under "_raw" instead of coercing to {}.
            tc = {**tc, "arguments": {"_raw": args_obj}}
        out.append(tc)
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
                    log_fn(
                        "planner.image.dispatch.args",
                        trace_id=trace_id,
                        args={k: args.get(k) for k in ("prompt", "width", "height", "steps", "cfg", "negative")},
                    )
                except Exception as ex:
                    logging.warning("args_prep.log_fn.error trace_id=%s err=%s", str(trace_id), str(ex), exc_info=True)
        enriched.append({**tc, "arguments": args})
    return enriched


