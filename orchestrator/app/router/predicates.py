from __future__ import annotations

import re


FILM_KEYS = ["film", "movie", "short film", "storyboard", "animatic", "shots", "scene list"]
RAG_KEYS = ["search", "lookup", "docs", "document", "kb", "retrieve", "rag"]
RESEARCH_KEYS = [
    "follow the money",
    "funding",
    "grant",
    "contract",
    "money map",
    "evidence ledger",
    "timeline",
    "investigate",
    "deep research",
]
IMAGE_KEYS = ["image", "picture", "render", "upscale", "edit", "inpaint", "outpaint", "png", "jpg"]
TTS_KEYS = ["tts", "voiceover", "narrate", "read this", "text to speech", "voice over"]
MUSIC_KEYS = ["music", "compose", "instrumental", "song", "track", "bpm", "stems", "variation"]


def _contains_any(text: str, keys: list[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keys)


def looks_like_film(req_text: str) -> bool:
    return _contains_any(req_text, FILM_KEYS)


def looks_like_rag(req_text: str) -> bool:
    # “search/lookup … about <X>” without money-map wording
    if _contains_any(req_text, RESEARCH_KEYS):
        return False
    return _contains_any(req_text, RAG_KEYS)


def looks_like_research(req_text: str) -> bool:
    return _contains_any(req_text, RESEARCH_KEYS)


def looks_like_image(req_text: str) -> bool:
    return _contains_any(req_text, IMAGE_KEYS)


def looks_like_tts(req_text: str) -> bool:
    return _contains_any(req_text, TTS_KEYS)


def looks_like_music(req_text: str) -> bool:
    return _contains_any(req_text, MUSIC_KEYS)


