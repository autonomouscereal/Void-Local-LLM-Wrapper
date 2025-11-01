from __future__ import annotations

from typing import Dict, Any


def write_story_bible(prompt: str, preset: dict) -> Dict[str, Any]:
    return {
        "meta": {"duration_s": int(preset.get("duration_s", 60)), "aspect": preset.get("aspect", "16:9"), "style": preset.get("style", "cinematic")},
        "prompt": prompt or "",
        "beats": ["intro", "conflict", "resolution"],
    }


def write_character_bible(prompt: str, preset: dict) -> Dict[str, Any]:
    return {"characters": [{"name": "Protagonist", "wardrobe": {}, "traits": []}]}


def merge_answers_into_bibles(story_bible: dict, char_bible: dict, answers: dict) -> None:
    from .clarifications import merge_answers_into_bibles as _merge
    _merge(story_bible, char_bible, answers)


