from __future__ import annotations

from typing import Optional, Dict
from ..refs.embeds import maybe_load


def load_face_embeds(ref_id: str) -> Optional[Dict]:
    return maybe_load(ref_id, "face_embeds")


def load_voice_embed(ref_id: str) -> Optional[Dict]:
    return maybe_load(ref_id, "voice_embed")


def load_music_embed(ref_id: str) -> Optional[Dict]:
    return maybe_load(ref_id, "music_embed")


