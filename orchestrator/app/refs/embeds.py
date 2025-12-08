from __future__ import annotations

import os
import json
from typing import List, Dict, Any

from .storage import ref_dir, write_atomic
from ..locks.builder import _blocking_voice_embedding


def _path(ref_id: str, name: str) -> str:
    return os.path.join(ref_dir(ref_id), f"{name}.json")


def compute_face_embeddings(ref_id: str, image_paths: list[str]) -> dict:
    emb = {"type": "face", "items": [{"path": p, "vec": None} for p in (image_paths or [])], "has_vectors": False}
    write_atomic(_path(ref_id, "face_embeds"), json.dumps(emb, ensure_ascii=False))
    return emb


def compute_voice_embedding(ref_id: str, sample_paths: list[str]) -> dict:
    """
    Compute and persist simple voice embeddings for a voice ref.

    - Uses the same blocking embedding routine as locks.builder so matching
      is consistent with voice QA.
    - Stores one vector per sample under items[].vec and flips has_vectors
      to True when at least one embedding is available.
    """
    items: List[Dict[str, Any]] = []
    has_vectors = False
    for p in (sample_paths or []):
        vec = None
        try:
            if isinstance(p, str) and p:
                vec = _blocking_voice_embedding(p)
        except Exception:
            # Embeddings are best-effort; callers can still use raw paths.
            vec = None
        if isinstance(vec, list):
            has_vectors = True
        items.append({"path": p, "vec": vec})
    emb: Dict[str, Any] = {"type": "voice", "items": items, "has_vectors": has_vectors}
    write_atomic(_path(ref_id, "voice_embed"), json.dumps(emb, ensure_ascii=False))
    return emb


def compute_music_embedding(ref_id: str, track_path: str | None, stem_paths: list[str] | None) -> dict:
    items = []
    if track_path:
        items.append({"path": track_path, "vec": None})
    for s in (stem_paths or []):
        items.append({"path": s, "vec": None})
    emb = {"type": "music", "items": items, "has_vectors": False}
    write_atomic(_path(ref_id, "music_embed"), json.dumps(emb, ensure_ascii=False))
    return emb


def maybe_load(ref_id: str, name: str) -> dict | None:
    p = _path(ref_id, name)
    try:
        from ..json_parser import JSONParser
        with open(p, "r", encoding="utf-8") as f:
            parser = JSONParser()
            # Embedding files are arbitrary dicts; coerce to generic mapping.
            sup = parser.parse_superset(f.read(), dict)
            data = sup["coerced"]
            return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None


