from __future__ import annotations

from typing import Any, Dict


def build_delta_plan(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse primitive scores into an accept/decline decision with deltas.
    Inputs shape (example):
      {"image": {"clip": 0.28}, "audio": {"clap": 0.31}, "locks": {"id_cos": 0.84}}
    """
    img = float(((scores.get("image") or {}).get("clip") or 0.0))
    idc = float(((scores.get("locks") or {}).get("id_cos") or 0.0))
    aud = float(((scores.get("audio") or {}).get("clap") or 0.0))
    accept = (img >= 0.30) and (idc >= 0.82 if idc > 0 else True) and (aud >= 0.30 if aud > 0 else True)
    if accept:
        return {"accept": True, "reasons": [], "deltas": [], "targets": {"clip_min": 0.30, "id_cos_min": 0.82, "clap_min": 0.30}}
    deltas = []
    reasons = []
    if img < 0.30:
        reasons.append("CLIP low")
        deltas.append({"engine": "image.edit", "op": "inpaint", "mask": "auto_small", "prompt": "+sharper, clearer subject"})
    if idc and idc < 0.82:
        reasons.append("ID lock drift")
        deltas.append({"engine": "image.edit", "op": "identity_boost", "prompt": "+stronger FaceID+ lock"})
    if aud and aud < 0.30:
        reasons.append("CLAP relevance low")
        deltas.append({"engine": "audio.variation", "op": "variation", "prompt": "+brighter, cleaner"})
    return {"accept": False, "reasons": reasons, "deltas": deltas, "targets": {"clip_min": 0.30, "id_cos_min": 0.82, "clap_min": 0.30}}


