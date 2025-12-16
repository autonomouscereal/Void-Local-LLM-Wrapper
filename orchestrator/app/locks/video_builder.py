from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import logging

import cv2  # type: ignore

from .builder import _save_image_bytes, _compute_face_embedding, _now_ts

log = logging.getLogger(__name__)


async def build_video_bundle(
    character_id: str,
    video_path: str,
    *,
    locks_root_dir: str,
    max_frames: int = 16,
) -> Dict[str, Any]:
    """
    Build a schema_version=2-style lock bundle from a reference video.

    This is intentionally simple and reuses the same embedding space and bundle
    shape as build_image_bundle:

    - Sample up to max_frames frames from the video.
    - For each frame, rely on InsightFace to detect the primary face and
      compute an embedding via _compute_face_embedding.
    - Aggregate all valid embeddings into a mean id_embedding.
    - Save frame snapshots under locks_root_dir/image/{character_id}.
    - Expose the first frame as image_path and the rest as extra_image_paths.
    """
    # Input validation is intentionally minimal here; invalid values will surface
    # as natural errors later in the pipeline rather than being guarded.
    if not isinstance(character_id, str) or not character_id.strip():
        character_id = str(character_id or "").strip()
    if not isinstance(video_path, str) or not video_path.strip():
        video_path = str(video_path or "").strip()

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_paths: List[str] = []
    try:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # Encode frame to PNG bytes for reuse of _save_image_bytes.
            ok_enc, buf = cv2.imencode(".png", frame)
            if not ok_enc:
                frame_idx += 1
                continue
            raw_bytes = buf.tobytes()
            suffix = f"video_ref_{_now_ts()}_{frame_idx}"
            path = _save_image_bytes(locks_root_dir, character_id, raw_bytes, suffix)
            saved_paths.append(path)
            frame_idx += 1
    finally:
        try:
            cap.release()
        except Exception:
            log.debug("video_builder: failed to release cv2.VideoCapture (non-fatal) video_path=%s", video_path, exc_info=True)

    # If no frames were saved, downstream indexing into saved_paths will fail
    # naturally instead of being guarded by an explicit ValueError here.

    # Compute embeddings for all saved frames and aggregate into a mean.
    embs: List[List[float]] = []
    for p in saved_paths:
        try:
            emb = await _compute_face_embedding(p)
        except Exception:
            emb = None
        if isinstance(emb, list):
            embs.append(emb)

    if not embs:
        # Fallback: still return a bundle but without an embedding so callers can
        # decide how to handle missing identity vectors.
        mean_emb: Optional[List[float]] = None
    else:
        dim = len(embs[0])
        acc = [0.0] * dim
        count = 0
        for vec in embs:
            if not isinstance(vec, list) or len(vec) != dim:
                continue
            for i, v in enumerate(vec):
                try:
                    acc[i] += float(v)
                except Exception:
                    acc[i] += 0.0
            count += 1
        mean_emb = [v / float(count) for v in acc] if count > 0 else None

    primary_path = saved_paths[0]
    extra_paths = saved_paths[1:]

    face_block: Dict[str, Any] = {
        "embedding": mean_emb,
        "mask": None,
        "image_path": primary_path,
        "strength": 0.75,
    }
    if extra_paths:
        face_block["extra_image_paths"] = extra_paths

    bundle: Dict[str, Any] = {
        "schema_version": 2,
        "character_id": character_id,
        "face": face_block,
        "pose": {},
        "style": {},
        "audio": {},
        "regions": {},
        "scene": {
            "background_embedding": None,
            "camera_style_tags": [],
            "lighting_tags": [],
            "lock_mode": "soft",
        },
    }
    return bundle


