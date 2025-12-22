from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os

from ..quality.eval.music_eval import eval_technical
from ..tracing.runtime import trace_event
from ..ref_library.storage import load_manifest


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")


def _map_upload_path(p: str) -> str:
    if not isinstance(p, str) or not p:
        return p
    if p.startswith("/uploads/"):
        return os.path.join(UPLOAD_DIR, p.split("/")[-1])
    if p.startswith("http") and "/uploads/" in p:
        return os.path.join(UPLOAD_DIR, p.split("/")[-1])
    return p


def build_music_profile(*, ref_ids: List[str] | None = None, audio_paths: List[str] | None = None) -> Dict[str, Any]:
    """
    Build a summary music profile from one or more music refs or raw audio paths.
    The profile is suitable for seeding the `lock_bundle.music.*` branch.
    """
    ref_ids = [r for r in (ref_ids or []) if isinstance(r, str) and r]
    audio_paths = [p for p in (audio_paths or []) if isinstance(p, str) and p]
    tracks: List[str] = []

    for rid in ref_ids:
        try:
            man = load_manifest(rid)
        except Exception:
            man = None
        if not man:
            continue
        if (man.get("files", {}).get("track") or {}).get("path"):
            tracks.append((man.get("files", {}).get("track") or {}).get("path"))
        for f in man.get("files", {}).get("stems", []) or []:
            if isinstance(f, dict):
                p = f.get("path")
                if isinstance(p, str) and p:
                    tracks.append(p)

    for p in audio_paths:
        tracks.append(_map_upload_path(p))

    metrics: List[Dict[str, Any]] = []
    for p in tracks:
        if not isinstance(p, str) or not p:
            continue
        try:
            m = eval_technical(p)
        except Exception:
            m = {}
        if isinstance(m, dict):
            m["path"] = p
            metrics.append(m)

    if not metrics:
        return {"ok": False, "profile": {}, "reason": "no_valid_tracks"}

    tempos = [m.get("tempo_bpm") for m in metrics if isinstance(m.get("tempo_bpm"), (int, float))]
    keys = [m.get("key") for m in metrics if isinstance(m.get("key"), str) and m.get("key")]
    genres = [m.get("genre") for m in metrics if isinstance(m.get("genre"), str) and m.get("genre")]
    emotions = [m.get("emotion") for m in metrics if isinstance(m.get("emotion"), str) and m.get("emotion")]

    def _range(vals: List[float]) -> Tuple[float, float] | None:
        if not vals:
            return None
        return float(min(vals)), float(max(vals))

    bpm_range = _range([float(t) for t in tempos]) if tempos else None
    key_histogram: Dict[str, int] = {}
    for k in keys:
        key_histogram[k] = key_histogram.get(k, 0) + 1

    style_tags: List[str] = []
    # Coarse tags from genre/emotion
    for g in genres:
        if g not in style_tags:
            style_tags.append(g)
    for e in emotions:
        if e not in style_tags:
            style_tags.append(e)

    profile: Dict[str, Any] = {
        "style_tags": style_tags,
        "bpm_range": bpm_range,
        "keys": key_histogram,
        "tracks": metrics,
    }

    # New canonical event name, but keep legacy-compatible payload shape.
    trace_event(
        "locks.music.build_profile",
        {
            "ref_ids": ref_ids,
            "tracks_count": len(tracks),
            "bpm_range": bpm_range,
            "keys_histogram": key_histogram,
            "style_tags": style_tags,
        },
    )
    return {"ok": True, "profile": profile}


