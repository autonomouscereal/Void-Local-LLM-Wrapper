from __future__ import annotations

from typing import Optional, Dict, Any


def resolve_music_lock(music_id: Optional[str], inline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Canonical resolver for music lock inputs used by music.* tools.
    """
    if inline and isinstance(inline, dict):
        return inline
    if music_id:
        return {"music_id": music_id}
    return {}


