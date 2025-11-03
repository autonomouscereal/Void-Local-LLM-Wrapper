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
IMAGE_KEYS = [
    "image", "picture", "photo", "art", "illustration", "sketch", "draw", "paint",
    "generate an image", "make an image", "make a picture", "render", "upscale", "edit",
    "inpaint", "outpaint", "png", "jpg", "jpeg", "webp"
]
TTS_KEYS = ["tts", "voiceover", "narrate", "read this", "text to speech", "voice over"]
MUSIC_KEYS = ["music", "compose", "instrumental", "song", "track", "bpm", "stems", "variation"]
CODE_KEYS = ["patch", "diff", "modify file", "add endpoint", "refactor", "unified diff"]
MATH_KEYS = [
    "integrate", "derivative", "differentiate", "limit", "solve", "simplify", "factor", "expand",
    "∫", "d/dx", "Σ", "sum", "product", "matrix", "det", "+", "-", "*", "/", "^", "sin(", "cos(", "tan(",
]


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


def looks_like_code_task(req_text: str) -> bool:
    return _contains_any(req_text, CODE_KEYS)
def looks_like_math(req_text: str) -> bool:
    return _contains_any(req_text, MATH_KEYS)



