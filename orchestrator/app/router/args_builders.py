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


def build_music_song_args(req_text: str) -> Dict[str, Any]:
    # Treat the incoming text as lyrics; style_tags empty by default
    return {"lyrics": req_text, "style_tags": [], "bpm": None, "key": None, "seed": None, "reference_song": None}


def build_musicgen_args(req_text: str) -> Dict[str, Any]:
    # Instrumental by default; melody lock can be attached by tools later
    return {"text": req_text, "melody_wav": None, "bpm": None, "key": None, "seed": None, "style_tags": [], "length_s": 30}


def build_sao_args(req_text: str) -> Dict[str, Any]:
    # Try to parse a duration hint; fallback to 8 seconds
    import re
    seconds = 8
    m = re.search(r"(\d{1,3})\s*(s|sec|secs|second|seconds)\b", (req_text or "").lower())
    if m:
        try:
            seconds = max(1, min(300, int(m.group(1))))
        except Exception:
            seconds = 8
    return {"text": req_text, "seconds": seconds, "bpm": None, "seed": None, "genre_tags": []}


def build_demucs_args(req_text: str) -> Dict[str, Any]:
    # The actual mix path should be supplied via attachments; here we just declare desired stems
    return {"mix_wav": None, "stems": ["vocals", "drums", "bass", "other"]}


def build_dsrvc_args(req_text: str) -> Dict[str, Any]:
    # Singing synthesis with optional melody; target voice provided via refs
    return {"lyrics": req_text, "notes_midi": None, "melody_wav": None, "target_voice_ref": None, "seed": None}


def build_math_args(req_text: str) -> Dict[str, Any]:
    return {"expr": req_text}


