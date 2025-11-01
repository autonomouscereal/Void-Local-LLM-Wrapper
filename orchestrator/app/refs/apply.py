from __future__ import annotations

from typing import Optional, List, Dict


def load_refs(ref_ids: Optional[List[str]], inline_refs: Optional[Dict]) -> Dict:
    """
    Minimal pass-through for now. In Step 19, wire to persistent refs registry.
    inline_refs can include: {"images":[], "faces":[], "video_frames":[], "pose":path, "depth":path, "char_id":str}
    """
    return inline_refs or {}


