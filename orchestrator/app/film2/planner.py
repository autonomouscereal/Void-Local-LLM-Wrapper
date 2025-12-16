from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def _as_dict(v: Any, *, name: str) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if v is None:
        log.warning("film2.planner %s missing; defaulting to {}", name)
        return {}
    log.warning("film2.planner %s invalid type=%s; defaulting to {}", name, type(v).__name__)
    return {}


def _as_str_list(v: Any, *, name: str) -> List[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        log.warning("film2.planner %s not list type=%s; dropping", name, type(v).__name__)
        return []
    out: List[str] = []
    dropped = 0
    for x in v:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
        else:
            dropped += 1
    if dropped:
        log.warning("film2.planner %s dropped=%d kept=%d", name, dropped, len(out))
    return out


def build_scenes(story_bible: dict) -> List[Dict]:
    sb = _as_dict(story_bible, name="story_bible")
    beats = _as_str_list(sb.get("beats"), name="story_bible.beats") or ["intro", "conflict", "resolution"]
    scenes = [{"id": f"sc{i:02d}", "beat": b, "loc": "INT", "time": "DAY"} for i, b in enumerate(beats, 1)]
    log.info("film2.planner.build_scenes beats=%d scenes=%d", len(beats), len(scenes))
    return scenes


def _character_names(char_bible: Any) -> List[str]:
    cb = _as_dict(char_bible, name="char_bible")
    chars_val = cb.get("characters")
    if chars_val is None:
        return []
    if not isinstance(chars_val, list):
        log.warning("film2.planner char_bible.characters not list type=%s; ignoring", type(chars_val).__name__)
        return []
    out: List[str] = []
    for c in chars_val:
        if isinstance(c, dict):
            nm = c.get("name")
            if isinstance(nm, str) and nm.strip():
                out.append(nm.strip())
        elif isinstance(c, str) and c.strip():
            out.append(c.strip())
    # Keep a stable, small cast per shot for downstream renderers.
    return out[:2]


def build_shots(scenes: List[Dict], char_bible: dict, coverage: str = "basic") -> List[Dict]:
    kinds = ["WIDE", "MEDIUM", "CLOSE"] if str(coverage or "").strip().lower() == "basic" else ["WIDE", "MEDIUM", "CLOSE", "INSERT"]
    shots: List[Dict] = []
    if not isinstance(scenes, list):
        log.warning("film2.planner.build_shots scenes not list type=%s; defaulting to []", type(scenes).__name__)
        scenes = []
    cast = _character_names(char_bible)
    i = 1
    invalid_scenes = 0
    for si, sc in enumerate(scenes, 1):
        if not isinstance(sc, dict):
            invalid_scenes += 1
            continue
        scene_id = sc.get("id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            scene_id = f"sc{si:02d}"
        for kind in kinds:
            shots.append(
                {
                    "id": f"sh{i:03d}",
                    "scene": scene_id,
                    "kind": kind,
                    "duration_ms": (1500 if kind != "INSERT" else 600),
                    "chars": list(cast),
                }
            )
            i += 1
    if invalid_scenes:
        log.warning("film2.planner.build_shots invalid_scenes=%d kept=%d", invalid_scenes, len(scenes) - invalid_scenes)
    log.info("film2.planner.build_shots scenes=%d kinds=%d shots=%d cast=%s coverage=%s", len(scenes), len(kinds), len(shots), cast, coverage)
    return shots


