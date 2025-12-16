from __future__ import annotations

from typing import Dict, Any
from .core import ablate
from void_envelopes import normalize_to_envelope


def post_ablate(raw_text: str, scope: str = "chat") -> Dict[str, Any]:
    env = normalize_to_envelope(raw_text)
    return {"ablated": ablate(env, scope_hint=scope)}


