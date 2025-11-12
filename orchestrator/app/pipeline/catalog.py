from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from app.tools_schema import get_builtin_tools_schema  # type: ignore


def build_allowed_tool_names() -> Set[str]:
	"""
	Collect allowed tool names from the route registry and built-in OpenAI-style schema.
	"""
	# Route registry
	try:
		from app.routes.tools import _REGISTRY as _TOOL_REG  # type: ignore
	except Exception:
		_TOOL_REG = {}
	allowed: Set[str] = set([str(k) for k in (_TOOL_REG or {}).keys()])
	# Builtins (OpenAI-style)
	try:
		builtins = get_builtin_tools_schema()
	except Exception:
		builtins = []
	for t in (builtins or []):
		try:
			fn = (t.get("function") or {})
			nm = fn.get("name")
			if nm:
				allowed.add(str(nm))
		except Exception:
			continue
	return allowed


def validate_tool_names(tool_calls: List[Dict[str, Any]], allowed: Set[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
	"""
	Filter out tool calls with unknown names. Return (valid_calls, unknown_names).
	"""
	valid: List[Dict[str, Any]] = []
	unknown: List[str] = []
	for tc in (tool_calls or []):
		try:
			nm = str((tc or {}).get("name") or "")
			if nm and (nm in allowed):
				valid.append(tc)
			elif nm:
				unknown.append(nm)
		except Exception:
			continue
	return valid, unknown


