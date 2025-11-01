from __future__ import annotations

from typing import List, Dict, Any
from .refs import inject_refs_into_final


def render_thumbnails(shots: List[Dict[str, Any]]):
    return [{"id": s.get("id"), "path": f"thumbnails/{s.get('id')}.png"} for s in (shots or [])]


def render_storyboards(shots: List[Dict[str, Any]], thumbs: List[Dict[str, Any]]):
    return [{"id": s.get("id"), "path": f"storyboards/{s.get('id')}.png", "seed": 1234, "style": "cinematic"} for s in (shots or [])]


def render_animatic(shots: List[Dict[str, Any]], sboards):
    return {"path": "animatic.mp4", "duration": sum(int(s.get("duration_ms", 0)) for s in (shots or [])) // 1000}


def render_final_shots(shots: List[Dict[str, Any]], refs: Dict[str, Dict[str, Any]]):
    outs: List[Dict[str, Any]] = []
    for s in (shots or []):
        s2 = inject_refs_into_final(dict(s), refs)
        outs.append({"id": s2.get("id"), "video": {"path": f"final_shots/{s2.get('id')}.mp4", "duration_ms": int(s2.get("duration_ms", 0))}})
    return outs


def assemble_final(final_shots: List[Dict[str, Any]]):
    return {"path": "final.mp4", "duration": sum(int(x.get("video", {}).get("duration_ms", 0)) for x in (final_shots or [])) // 1000}


