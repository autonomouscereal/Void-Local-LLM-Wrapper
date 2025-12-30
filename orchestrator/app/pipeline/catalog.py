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
    Normalize tool calls (fix name/tool_name format) and identify unknown tools.
    This is a validation/fixing step, NOT a blocker - all tool calls are returned.
    Returns (normalized_calls, unknown_names) where normalized_calls includes ALL calls.
    """
    normalized: List[Dict[str, Any]] = []
    unknown: List[str] = []
    for tc in (tool_calls or []):
        try:
            if not isinstance(tc, dict):
                # Skip non-dict entries but keep them in the output for transparency
                normalized.append(tc)
                continue
            
            # Normalize: ensure both "tool_name" and "name" are set for compatibility
            # Check both "tool_name" (from planner) and "name" (OpenAI format)
            tool_name = (tc.get("tool_name") or tc.get("name") or "").strip()
            if not tool_name:
                # No tool name found - keep as-is but mark as unknown
                normalized.append(tc)
                unknown.append("(no_name)")
                continue
            
            # Normalize the tool call: ensure both fields are set
            normalized_tc = dict(tc)
            normalized_tc["tool_name"] = tool_name
            normalized_tc["name"] = tool_name  # Also set "name" for OpenAI compatibility
            
            # Check if it's in the allowed set (for logging only, not filtering)
            if tool_name not in allowed:
                unknown.append(tool_name)
            
            normalized.append(normalized_tc)
        except Exception:
            # On any error, keep the original tool call
            normalized.append(tc)
            continue
    return normalized, unknown
