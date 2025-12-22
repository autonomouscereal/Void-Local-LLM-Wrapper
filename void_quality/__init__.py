from __future__ import annotations

"""
void_quality: shared quality/metrics/scoring configuration helpers.

This package is intended to be copied into service containers (like `void_json`
and `void_envelopes`) so thresholds and shared quality defaults stay consistent
across orchestrator + satellite services.
"""

from .thresholds import (  # noqa: F401
    get_film2_quality_thresholds,
    get_lock_quality_presets,
    get_music_acceptance_thresholds,
    get_quality_preset,
    lock_quality_thresholds,
)

__all__ = [
    "get_lock_quality_presets",
    "get_quality_preset",
    "lock_quality_thresholds",
    "get_music_acceptance_thresholds",
    "get_film2_quality_thresholds",
]




