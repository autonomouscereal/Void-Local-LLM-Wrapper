from __future__ import annotations

from typing import Optional, Dict


def resolve_voice_lock(voice_id: Optional[str], inline: Optional[Dict]) -> Dict:
    # Future: load from a store by voice_id. For now, pass inline or id stub.
    if inline and isinstance(inline, dict):
        return inline
    if voice_id:
        return {"voice_id": voice_id}
    return {}


