from __future__ import annotations

from typing import Dict, Any


def extract_refs_from_storyboards(sboards_dir) -> Dict[str, Dict[str, Any]]:
    """Build per-shot reference pack: pose/depth/seed/style."""
    refs: Dict[str, Dict[str, Any]] = {}
    for sb in (sboards_dir or []):
        if not isinstance(sb, dict):
            continue
        sid = sb.get("id")
        if not sid:
            continue
        refs[sid] = {
            "pose": sb.get("pose"),
            "depth": sb.get("depth"),
            "seed": sb.get("seed"),
            "style": sb.get("style"),
        }
    return refs


def inject_refs_into_final(shot: Dict[str, Any], refs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    r = refs.get(shot.get("id"), {})
    rp = dict(shot.get("render_params") or {})
    for k, v in r.items():
        if v is not None:
            rp[k] = v
    shot["render_params"] = rp
    return shot


