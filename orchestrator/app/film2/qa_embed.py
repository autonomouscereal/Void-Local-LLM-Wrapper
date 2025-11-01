from __future__ import annotations

from typing import Dict, Any, Optional


def _cos(a, b) -> float:
    if not a or not b:
        return 1.0
    try:
        import math
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a))
        db = math.sqrt(sum(y * y for y in b))
        return float(num) / (da * db + 1e-9)
    except Exception:
        return 1.0


def face_similarity(storyboard_meta: Dict[str, Any], final_meta: Dict[str, Any], face_ref_embed) -> float:
    sb_vec = (storyboard_meta or {}).get("face_vec")
    fin_vec = (final_meta or {}).get("face_vec")
    return _cos(sb_vec, fin_vec if fin_vec else face_ref_embed)


def voice_similarity(vo_meta: Dict[str, Any], voice_ref_embed) -> float:
    return _cos((vo_meta or {}).get("voice_vec"), voice_ref_embed)


def music_similarity(cue_meta: Dict[str, Any], motif_embed) -> float:
    return _cos((cue_meta or {}).get("motif_vec"), motif_embed)


