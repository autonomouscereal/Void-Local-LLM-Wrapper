from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, Optional

from void_json.json_parser import JSONParser


def _load_json_data(rel_path: str) -> Dict[str, Any]:
    # Package data loader: uses importlib.resources so it works inside containers
    # without relying on filesystem-relative paths.
    data = resources.files(__package__).joinpath("data").joinpath(rel_path).read_bytes()
    parser = JSONParser()
    obj = parser.parse(data.decode("utf-8", errors="replace"), {})
    return obj if isinstance(obj, dict) else {}


@lru_cache(maxsize=1)
def get_lock_quality_presets() -> Dict[str, Dict[str, Any]]:
    """
    Canonical lock/quality presets shared across orchestrator and services.

    This replaces ad-hoc copies of QUALITY_PRESETS scattered around the repo.
    """
    raw = _load_json_data("lock_quality_presets.json")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = dict(v)
    return out


def get_quality_preset(name: Optional[str]) -> Dict[str, Any]:
    presets = get_lock_quality_presets()
    key = (name or "standard").strip().lower() or "standard"
    return presets.get(key, presets.get("standard", {}))


def lock_quality_thresholds(profile: Optional[str]) -> Dict[str, float]:
    """
    Return the minimal acceptance thresholds for locks/scoring (face/regions/scene/audio).
    """
    preset = get_quality_preset(profile)

    def _f(key: str, default: float) -> float:
        v = preset.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    return {
        "face_min": _f("face_min", 0.8),
        "region_shape_min": _f("region_shape_min", 0.8),
        "region_texture_min": _f("region_texture_min", 0.8),
        "scene_min": _f("scene_min", 0.8),
        "voice_min": _f("voice_min", 0.85),
        "tempo_min": _f("tempo_min", 0.85),
        "key_min": _f("key_min", 0.85),
        "stem_balance_min": _f("stem_balance_min", 0.8),
        "lyrics_min": _f("lyrics_min", 0.85),
    }


@lru_cache(maxsize=1)
def get_music_acceptance_thresholds() -> Dict[str, float]:
    """
    Canonical acceptance thresholds for music evaluation.

    Replaces orchestrator-local review/acceptance_audio.json reads.
    """
    raw = _load_json_data("music_acceptance.json")
    music = raw.get("music") if isinstance(raw.get("music"), dict) else {}
    overall_min = float(music.get("overall_quality_min"))
    fit_min = float(music.get("fit_score_min"))
    return {"overall_quality_min": overall_min, "fit_score_min": fit_min}


def get_film2_quality_thresholds(env: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """
    Canonical Film2 QA thresholds.

    Defaults come from package data, but env vars can override for deployment tuning:
      - FILM2_QA_T_FACE
      - FILM2_QA_T_VOICE
      - FILM2_QA_T_MUSIC
    """
    raw = _load_json_data("film2_quality.json")
    base = raw.get("thresholds") if isinstance(raw.get("thresholds"), dict) else {}
    out = {
        "face": float(base.get("face", 0.85)),
        "voice": float(base.get("voice", 0.85)),
        "music": float(base.get("music", 0.80)),
    }
    e = env if isinstance(env, dict) else dict(os.environ)

    def _override(key: str, env_key: str) -> None:
        if env_key not in e:
            return
        try:
            out[key] = float(str(e.get(env_key) or "").strip())
        except Exception:
            # Keep default from package data
            return

    _override("face", "FILM2_QA_T_FACE")
    _override("voice", "FILM2_QA_T_VOICE")
    _override("music", "FILM2_QA_T_MUSIC")
    return out


__all__ = [
    "get_lock_quality_presets",
    "get_quality_preset",
    "lock_quality_thresholds",
    "get_music_acceptance_thresholds",
    "get_film2_quality_thresholds",
]




