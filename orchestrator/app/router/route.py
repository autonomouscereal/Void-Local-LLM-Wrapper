from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Dict, Any
# predicates removed; AI planner decides all tool choices
# args builders are not used here; planner decides and downstream tools normalize


@dataclass
class RouteDecision:
    kind: Literal["tool", "planner"]
    tool: str | None
    args: Dict[str, Any] | None
    reason: str


def _last_user_text(req: Dict[str, Any]) -> str:
    try:
        msgs = req.get("messages") or []
        for m in reversed(msgs):
            if (m.get("role") == "user") and isinstance(m.get("content"), str) and m.get("content").strip():
                return m.get("content").strip()
    except Exception:
        return ""
    return ""


def route_for_request(req: Dict[str, Any]) -> RouteDecision:
    # AI planner powers all decisions; fallback is always 'planner'
    return RouteDecision(kind="planner", tool=None, args=None, reason="planner-only")


