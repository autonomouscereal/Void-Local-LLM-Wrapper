from __future__ import annotations

from typing import List, Dict


def collect_one_shot(prompt: str, preset: dict) -> List[Dict[str, str]]:
    """Return a fixed list of clarifying questions (only once) if critical fields are missing."""
    qs: List[Dict[str, str]] = []
    if "duration_s" not in (preset or {}):
        qs.append({"key": "duration_s", "q": "Target duration (sec)?", "default": 60})
    if "aspect" not in (preset or {}):
        qs.append({"key": "aspect", "q": "Aspect? (16:9 / 9:16 / 2.39)", "default": "16:9"})
    if "style" not in (preset or {}):
        qs.append({"key": "style", "q": "Visual style?", "default": "cinematic"})
    qs.append({"key": "characters", "q": "Main character(s)? (comma-separated)", "default": "Protagonist"})
    return qs


def merge_answers_into_bibles(story_bible: dict, char_bible: dict, answers: dict) -> None:
    story_bible.setdefault("meta", {}).update({
        "duration_s": int((answers or {}).get("duration_s", story_bible.get("meta", {}).get("duration_s", 60))),
        "aspect": (answers or {}).get("aspect", story_bible.get("meta", {}).get("aspect", "16:9")),
        "style": (answers or {}).get("style", story_bible.get("meta", {}).get("style", "cinematic")),
    })
    names = [x.strip() for x in str((answers or {}).get("characters", "")).split(",") if x.strip()]
    if names:
        char_bible["characters"] = [{"name": n, "wardrobe": {}, "traits": []} for n in names]


