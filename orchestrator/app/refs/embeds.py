from __future__ import annotations

import os
import json
from .storage import ref_dir, write_atomic


def _path(ref_id: str, name: str) -> str:
    return os.path.join(ref_dir(ref_id), f"{name}.json")


def compute_face_embeddings(ref_id: str, image_paths: list[str]) -> dict:
    emb = {"type": "face", "items": [{"path": p, "vec": None} for p in (image_paths or [])]}
    write_atomic(_path(ref_id, "face_embeds"), json.dumps(emb, ensure_ascii=False))
    return emb


def compute_voice_embedding(ref_id: str, sample_paths: list[str]) -> dict:
    emb = {"type": "voice", "items": [{"path": p, "vec": None} for p in (sample_paths or [])]}
    write_atomic(_path(ref_id, "voice_embed"), json.dumps(emb, ensure_ascii=False))
    return emb


def compute_music_embedding(ref_id: str, track_path: str | None, stem_paths: list[str] | None) -> dict:
    items = []
    if track_path:
        items.append({"path": track_path, "vec": None})
    for s in (stem_paths or []):
        items.append({"path": s, "vec": None})
    emb = {"type": "music", "items": items}
    write_atomic(_path(ref_id, "music_embed"), json.dumps(emb, ensure_ascii=False))
    return emb


def maybe_load(ref_id: str, name: str) -> dict | None:
    p = _path(ref_id, name)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


