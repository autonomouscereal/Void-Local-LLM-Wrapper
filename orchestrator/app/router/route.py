from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any
from .predicates import (
    looks_like_film,
    looks_like_rag,
    looks_like_research,
    looks_like_image,
    looks_like_tts,
    looks_like_music,
)
from .args_builders import (
    build_film_args,
    build_rag_args,
    build_research_args,
    build_image_args,
    build_tts_args,
    build_music_args,
)


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
        pass
    return ""


def route_for_request(req: Dict[str, Any]) -> RouteDecision:
    t = _last_user_text(req)
    if looks_like_film(t):
        return RouteDecision(kind="tool", tool="film.run", args=build_film_args(t), reason="film-intent")
    if looks_like_research(t):
        return RouteDecision(kind="tool", tool="research.run", args=build_research_args(t), reason="research-intent")
    if looks_like_rag(t):
        return RouteDecision(kind="tool", tool="rag_search", args=build_rag_args(t), reason="rag-intent")
    if looks_like_image(t):
        return RouteDecision(kind="tool", tool="image.dispatch", args=build_image_args(t), reason="image-intent")
    if looks_like_tts(t):
        return RouteDecision(kind="tool", tool="tts_speak", args=build_tts_args(t), reason="tts-intent")
    if looks_like_music(t):
        return RouteDecision(kind="tool", tool="music.dispatch", args=build_music_args(t), reason="music-intent")
    return RouteDecision(kind="planner", tool=None, args=None, reason="fallback-planner")


