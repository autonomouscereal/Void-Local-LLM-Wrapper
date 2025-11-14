from __future__ import annotations

LOCK_BUNDLE_SCHEMA = {
    "schema_version": 2,
    "character_id": "string",
    "face": {
        "embedding": "base64 or list[float]",
        "mask": "optional path or base64",
        "image_path": "optional stored path",
        "strength": 0.75,
    },
    "pose": {
        "skeleton": "optional path/json",
        "strength": 0.7,
    },
    "style": {
        "prompt_tags": ["example tag"],
        "palette": {"primary": "#D00000", "accent": "#111111"},
    },
    "audio": {
        "voice_embedding": "optional base64 or list[float]",
        "timbre_tags": ["raspy", "female", "aggressive"],
        "tempo_bpm": 0.0,
        "tempo_lock_mode": "hard|soft|off",
        "key": "C minor",
        "key_lock_mode": "hard|soft|off",
        "stem_profile": {},
        "stem_lock_mode": "hard|soft|off",
        "lyrics_segments": [],
    },
    "regions": {},
    "scene": {
        "background_embedding": None,
        "camera_style_tags": [],
        "lighting_tags": [],
        "lock_mode": "soft",
    },
}

