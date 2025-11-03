from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Dict, Any
from .predicates import (
    looks_like_film,
    looks_like_rag,
    looks_like_research,
    looks_like_image,
    looks_like_tts,
    looks_like_music,
    looks_like_code_task,
)
from .args_builders import (
    build_film_args,
    build_rag_args,
    build_research_args,
    build_image_args,
    build_tts_args,
    build_music_args,
    build_yue_args,
    build_musicgen_args,
    build_sao_args,
    build_demucs_args,
    build_dsrvc_args,
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
    if looks_like_code_task(t):
        return RouteDecision(kind="tool", tool="code.super_loop", args={"task": t, "repo_root": os.getenv("REPO_ROOT", "/workspace")}, reason="code-intent")
    if looks_like_film(t):
        # Heuristic: if video parameters imply heavy render, prefer Hunyuan path downstream
        return RouteDecision(kind="tool", tool="film.run", args=build_film_args(t), reason="film-intent")
    if looks_like_research(t):
        return RouteDecision(kind="tool", tool="research.run", args=build_research_args(t), reason="research-intent")
    if looks_like_rag(t):
        return RouteDecision(kind="tool", tool="rag_search", args=build_rag_args(t), reason="rag-intent")
    if looks_like_image(t):
        return RouteDecision(kind="tool", tool="image.dispatch", args=build_image_args(t), reason="image-intent")
    if looks_like_tts(t):
        return RouteDecision(kind="tool", tool="tts.speak", args=build_tts_args(t), reason="tts-intent")
    if looks_like_music(t):
        # Fine-grained deterministic routing among music/audio tools
        tl = (t or "").lower()
        import re
        if any(k in tl for k in ["lyrics", "chorus", "verse"]):
            return RouteDecision(kind="tool", tool="music.song.yue", args=build_yue_args(t), reason="music-lyrics")
        if any(k in tl for k in ["duration", "seconds", "sec", "sfx", "sound effect", "foley clip"]):
            return RouteDecision(kind="tool", tool="music.timed.sao", args=build_sao_args(t), reason="music-duration")
        if any(k in tl for k in ["stems", "separate", "split tracks"]):
            return RouteDecision(kind="tool", tool="audio.stems.demucs", args=build_demucs_args(t), reason="music-stems")
        if any(k in tl for k in ["sing", "timbre", "convert voice", "voice convert", "rvc"]):
            return RouteDecision(kind="tool", tool="voice.sing.diffsinger.rvc", args=build_dsrvc_args(t), reason="music-sing")
        if any(k in tl for k in ["melody", "hum", "humming", "whistle"]):
            return RouteDecision(kind="tool", tool="music.melody.musicgen", args=build_musicgen_args(t), reason="music-melody")
        # Default to full song (YuE) for best-quality audio by default
        return RouteDecision(kind="tool", tool="music.song.yue", args=build_yue_args(t), reason="music-default-yue")
    return RouteDecision(kind="planner", tool=None, args=None, reason="fallback-planner")


