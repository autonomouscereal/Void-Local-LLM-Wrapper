from __future__ import annotations

from typing import Dict, Any


def build_film_args(req_text: str) -> Dict[str, Any]:
    return {
        "prompt": (req_text or "").strip(),
        "clarifications": None,
        "preset": {"aspect": "16:9", "duration_s": 60, "style": "cinematic"},
    }


def build_rag_args(req_text: str) -> Dict[str, Any]:
    return {"query": (req_text or "").strip(), "k": 8}


def build_research_args(req_text: str) -> Dict[str, Any]:
    return {"query": (req_text or "").strip(), "scope": "public"}


def build_image_args(req_text: str) -> Dict[str, Any]:
    t = (req_text or "").lower()
    if "upscale" in t:
        return {"mode": "upscale", "image_ref": None, "scale": 2}
    if any(k in t for k in ["edit", "inpaint", "outpaint"]):
        return {"mode": "edit", "image_ref": None, "mask": None, "prompt": req_text}
    return {"mode": "gen", "prompt": req_text, "size": "1024x1024", "seed": None}


def build_tts_args(req_text: str) -> Dict[str, Any]:
    return {"text": req_text, "voice": "narrator", "rate": 1.0, "pitch": 0.0, "sample_rate": 22050}


def build_music_args(req_text: str) -> Dict[str, Any]:
    return {"prompt": req_text, "bpm": 120, "length_s": 30, "structure": ["intro", "verse", "outro"], "seed": None}


