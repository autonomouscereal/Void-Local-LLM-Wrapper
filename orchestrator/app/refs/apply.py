from __future__ import annotations

from typing import Optional, List, Dict
from .storage import load_manifest


def load_refs(ref_ids: Optional[List[str]], inline_refs: Optional[Dict]) -> Dict:
    """
    Merge inline refs with any referenced registry IDs into a single pack.
    For images: returns {"images":[...]} etc.
    """
    pack: Dict[str, any] = dict(inline_refs or {})
    for rid in (ref_ids or []):
        man = load_manifest(rid)
        if not man:
            continue
        if man.get("kind") == "image":
            imgs = [f.get("path") for f in man.get("files", {}).get("images", [])]
            if imgs:
                pack.setdefault("images", [])
                pack["images"].extend(imgs)
        if man.get("kind") == "voice":
            vs = [f.get("path") for f in man.get("files", {}).get("voice_samples", [])]
            if vs:
                pack.setdefault("voice_samples", [])
                pack["voice_samples"].extend(vs)
        if man.get("kind") == "music":
            tr = (man.get("files", {}).get("track") or {}).get("path")
            st = [f.get("path") for f in man.get("files", {}).get("stems", [])]
            if tr:
                pack["track"] = tr
            if st:
                pack.setdefault("stems", [])
                pack["stems"].extend(st)
    return pack


