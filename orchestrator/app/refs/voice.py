from __future__ import annotations

from typing import Optional, Dict
from .storage import load_manifest


def resolve_voice_lock(voice_id: Optional[str], inline: Optional[Dict]) -> Dict:
    if inline and isinstance(inline, dict):
        return inline
    if voice_id:
        man = load_manifest(voice_id)
        if man and man.get("kind") == "voice":
            return {"voice_samples": [f.get("path") for f in man.get("files", {}).get("voice_samples", [])]}
        return {"voice_id": voice_id}
    return {}


