from __future__ import annotations

import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple

from .storage import ref_dir, write_atomic
from ..locks.builder import _blocking_voice_embedding

import httpx  # type: ignore

from ..json_parser import JSONParser


def _faceid_base_url() -> str:
    return (os.environ.get("FACEID_API_URL") or "").strip()


def _faceid_embed_sync(*, image_path: str, model_name: Optional[str] = None, max_faces: int = 16) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Sync FaceID call used by ref_library codepaths (these helpers are synchronous and
    may be invoked inside non-async utilities).
    """
    base = _faceid_base_url()
    if not base:
        return ([], None)
    payload: Dict[str, Any] = {"image_path": image_path, "model_name": model_name, "max_faces": int(max_faces)}
    with httpx.Client(timeout=None, follow_redirects=True) as client:
        r = client.post(base.rstrip("/") + "/embed", json=payload)
    parser = JSONParser()
    env = parser.parse(
        r.text or "",
        {
            "ok": bool,
            "result": {
                "faces": [{"embedding": list, "det_score": float, "bbox": list}],
                "model": str,
            },
            "error": dict,
        },
    )
    if not isinstance(env, dict) or not bool(env.get("ok")):
        return ([], None)
    res = env.get("result") if isinstance(env.get("result"), dict) else {}
    faces = res.get("faces") if isinstance(res, dict) else []
    out_faces: List[Dict[str, Any]] = [f for f in faces if isinstance(f, dict)] if isinstance(faces, list) else []
    model = res.get("model") if isinstance(res, dict) and isinstance(res.get("model"), str) else None
    return (out_faces, model)


def _path(ref_id: str, name: str) -> str:
    return os.path.join(ref_dir(ref_id), f"{name}.json")


def compute_face_embeddings(ref_id: str, image_paths: list[str]) -> dict:
    """
    Compute and persist face embeddings for an image reference manifest.

    Uses the faceid service for canonical InsightFace embeddings, so identity
    comparisons stay in one embedding space without local model duplication.
    """
    items: List[Dict[str, Any]] = []
    has_vectors = False
    model = None
    for p in (image_paths or []):
        best_vec = None
        faces = []
        try:
            if isinstance(p, str) and p:
                faces, model_used = _faceid_embed_sync(image_path=p, model_name=model, max_faces=16)
                if model_used and not model:
                    model = model_used
                best_vec = faces[0].get("embedding") if faces else None
        except Exception as ex:
            # Don't fail the whole pack; persist partials but record the error.
            best_vec = None
            faces = []
            logging.getLogger(__name__).exception("ref_library.compute_face_embeddings: failed path=%s error=%s", p, ex)
        if isinstance(best_vec, list):
            has_vectors = True
        items.append({"path": p, "vec": best_vec, "faces": faces})
    emb: Dict[str, Any] = {
        "type": "face",
        "items": items,
        "has_vectors": has_vectors,
        "model": model,
    }
    write_atomic(_path(ref_id, "face_embeds"), json.dumps(emb, ensure_ascii=False))
    return emb


def compute_voice_embedding(ref_id: str, sample_paths: list[str]) -> dict:
    """
    Compute and persist simple voice embeddings for a voice reference.

    - Uses the same blocking embedding routine as locks.builder so matching is consistent.
    """
    items: List[Dict[str, Any]] = []
    has_vectors = False
    for p in (sample_paths or []):
        vec = None
        try:
            if isinstance(p, str) and p:
                vec = _blocking_voice_embedding(p)
        except Exception:
            vec = None
        if isinstance(vec, list):
            has_vectors = True
        items.append({"path": p, "vec": vec})
    emb: Dict[str, Any] = {"type": "voice", "items": items, "has_vectors": has_vectors}
    write_atomic(_path(ref_id, "voice_embed"), json.dumps(emb, ensure_ascii=False))
    return emb


def compute_music_embedding(ref_id: str, track_path: str | None, stem_paths: list[str] | None) -> dict:
    # Placeholder: kept for back-compat; music embeddings not yet implemented.
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
        with open(p, "r", encoding="utf-8") as f:
            parser = JSONParser()
            # Embedding files are arbitrary dicts; coerce to generic mapping.
            data = parser.parse(f.read(), {})
            return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None


