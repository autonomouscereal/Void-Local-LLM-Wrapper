from __future__ import annotations

from typing import Dict, List


# Minimal tool catalog for shape validation only (not routing)
REQUIRED_INPUTS: Dict[str, List[str]] = {
    # Image/Comfy
    "image.dispatch": ["prompt"],  # negative_prompt optional; locks optional
    "image.pose.map": ["image_id"],
    "image.edge.map": ["image_id"],
    "image.depth.map": ["image_id"],
    # Film-2 (single entry point for all video)
    # Keep the schema minimal and forgiving; Film-2 will expand/validate shots internally.
    "film2.run": ["prompt"],
    # Audio/Music
    "audio.music.generate": ["style", "duration_s"],
    "audio.vocal.synthesis": ["lyrics"],
    "audio.master": ["audio_id"],
    "audio.stems.demucs": ["audio_id"],
    "locks.build_image_bundle": ["character_id", "image_url"],
    "locks.build_audio_bundle": ["character_id", "audio_url"],
    "locks.get_bundle": ["character_id"],
    "locks.build_region_locks": ["character_id", "image_url"],
    "locks.update_region_modes": ["character_id", "updates"],
    "locks.update_audio_modes": ["character_id", "update"],
    "http.request": ["url", "method"],
}


