from __future__ import annotations

from typing import Optional, Dict


def resolve_music_lock(music_id: Optional[str], inline: Optional[Dict]) -> Dict:
    if inline and isinstance(inline, dict):
        return inline
    if music_id:
        return {"music_id": music_id}
    return {}


