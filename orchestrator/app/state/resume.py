from __future__ import annotations

from typing import Tuple, Dict, Any, List, Set
from .checkpoints import read_all, last_event
from .ids import short_hash


def reconstruct_window_state(root: str, conversation_id: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Build a minimal state for WindowedSolver: anchor_text, candidates, entities.
    Also returns partials so we can continue CONT/HALT.
    """
    evs = read_all(root, conversation_id)
    partials: List[str] = []
    entities: Set[str] = set()
    candidates: List[str] = []
    anchor = ""
    for e in evs:
        k = e.get("kind")
        d = e.get("data", {}) or {}
        if k == "window_step" and isinstance(d.get("text"), str):
            partials.append(d.get("text"))
        if k == "entities" and isinstance(d.get("list"), list):
            for it in d.get("list"):
                if isinstance(it, str):
                    entities.add(it)
        if k == "anchor" and isinstance(d.get("text"), str):
            anchor = d.get("text")
        if k == "retrieved" and isinstance(d.get("chunks"), list):
            for c in d.get("chunks"):
                if isinstance(c, dict):
                    txt = c.get("text") or c.get("chunk") or ""
                    if isinstance(txt, str) and txt.strip():
                        candidates.append(txt)
    # deduplicate candidates deterministically
    seen = set()
    dedup: List[str] = []
    for c in candidates:
        h = short_hash(c.encode("utf-8"))
        if h in seen:
            continue
        seen.add(h)
        dedup.append(c)
    state = {"anchor_text": anchor, "entities": sorted(entities), "candidates": dedup}
    return state, partials


def reconstruct_film_checkpoint(root: str, conversation_id: str) -> Dict[str, Any]:
    ev = last_event(root, conversation_id, "phase") or {"data": {"phase": 0}}
    phase = int(ev.get("data", {}).get("phase", 0))
    shots_done: Set[str] = set()
    for e in read_all(root, conversation_id):
        if e.get("kind") == "shot_done":
            shot_id = (e.get("data", {}) or {}).get("shot_id")
            shots_done.add(shot_id)
    return {"phase": phase, "shots_done": sorted(shots_done)}


