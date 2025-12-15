from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def build_allowed_tool_names(builtins_provider: Optional[Callable[[], List[Dict[str, Any]]]] = None) -> Set[str]:
    """
    Collect allowed tool names from the built-in OpenAI-style schema.
    """
    allowed: Set[str] = set()
    # Builtins (OpenAI-style)
    builtins: List[Dict[str, Any]] = []
    if callable(builtins_provider):
        try:
            builtins = builtins_provider() or []
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
