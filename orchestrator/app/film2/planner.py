from __future__ import annotations

from typing import List, Dict


def build_scenes(story_bible: dict) -> List[Dict]:
    beats = story_bible.get("beats") or ["intro", "conflict", "resolution"]
    return [{"id": f"sc{i:02d}", "beat": b, "loc": "INT", "time": "DAY"} for i, b in enumerate(beats, 1)]


def build_shots(scenes: List[Dict], char_bible: dict, coverage: str = "basic") -> List[Dict]:
    kinds = ["WIDE", "MEDIUM", "CLOSE"] if coverage == "basic" else ["WIDE", "MEDIUM", "CLOSE", "INSERT"]
    shots: List[Dict] = []
    i = 1
    for sc in scenes:
        for kind in kinds:
            shots.append({
                "id": f"sh{i:03d}",
                "scene": sc["id"],
                "kind": kind,
                "duration_ms": (1500 if kind != "INSERT" else 600),
                "chars": [c.get("name") for c in (char_bible.get("characters") or [])][:2],
            })
            i += 1
    return shots


