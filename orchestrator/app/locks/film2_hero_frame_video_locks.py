from __future__ import annotations

"""
Film2: hero-frame selection + lock scoring for videos.

This is the canonical implementation used by `film2.run` to:
- sample deterministic frames from a generated video
- compute *all* lock-domain scores on each frame (face/pose/style/scene/regions)
- choose the best hero frame for downstream lock bundle updates + QA + distillation

Terminology:
- "hero frame" = best single still frame representing the shot according to lock adherence.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore

from .image_lock_scoring import (
    compute_multiface_identity_scores,
    compute_pose_similarity,
    compute_region_scores,
    compute_scene_score,
    compute_style_similarity,
)
from .runtime import bundle_to_image_locks


def _abs_upload_path(upload_dir: str, maybe_path: str) -> str:
    if not isinstance(maybe_path, str) or not maybe_path:
        return ""
    if maybe_path.startswith("/workspace/"):
        return maybe_path
    if maybe_path.startswith("/uploads/"):
        return "/workspace" + maybe_path
    if maybe_path.startswith("/workspace/uploads/"):
        return maybe_path
    if not maybe_path.startswith("/"):
        return os.path.join(upload_dir, maybe_path)
    return maybe_path


def _extract_scene_ref(lock_bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    vis = lock_bundle.get("visual")
    if isinstance(vis, dict) and isinstance(vis.get("scene"), dict):
        return vis.get("scene")
    scene = lock_bundle.get("scene")
    return scene if isinstance(scene, dict) else None


def _derive_face_refs(lock_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert bundle_to_image_locks()["faces"] into compute_multiface_identity_scores face_refs.
    """
    locks_payload = bundle_to_image_locks(lock_bundle) if isinstance(lock_bundle, dict) else {}
    faces = locks_payload.get("faces") if isinstance(locks_payload.get("faces"), list) else []
    face_refs: List[Dict[str, Any]] = []
    for f in faces:
        if not isinstance(f, dict):
            continue
        emb = f.get("embedding")
        if not isinstance(emb, list):
            continue
        entity_id = f.get("entity_id")
        region = f.get("region") if isinstance(f.get("region"), dict) else {}
        # Map weight→priority so "harder" faces get assigned first.
        w = f.get("weight")
        priority = int(float(w) * 100.0) if isinstance(w, (int, float)) else 0
        face_refs.append(
            {
                "entity_id": str(entity_id) if isinstance(entity_id, (str, int)) else f"face_{len(face_refs)}",
                "priority": priority,
                "role": f.get("role"),
                "embedding": emb,
                "region": region,
            }
        )
    return face_refs


def _derive_region_refs(lock_bundle: Dict[str, Any]) -> Dict[str, Any]:
    locks_payload = bundle_to_image_locks(lock_bundle) if isinstance(lock_bundle, dict) else {}
    return locks_payload.get("regions") if isinstance(locks_payload.get("regions"), dict) else {}


def _derive_style_ref(lock_bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    locks_payload = bundle_to_image_locks(lock_bundle) if isinstance(lock_bundle, dict) else {}
    style_tags = locks_payload.get("style_tags") if isinstance(locks_payload.get("style_tags"), list) else []
    style_palette = locks_payload.get("style_palette") if isinstance(locks_payload.get("style_palette"), dict) else {}
    if not style_tags and not style_palette:
        return None
    return {"style_tags": style_tags, "palette": style_palette}


def _derive_pose_ref(lock_bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    locks_payload = bundle_to_image_locks(lock_bundle) if isinstance(lock_bundle, dict) else {}
    poses = locks_payload.get("poses") if isinstance(locks_payload.get("poses"), list) else []
    if not poses:
        return None
    first = poses[0] if isinstance(poses[0], dict) else None
    if not isinstance(first, dict):
        return None
    if not isinstance(first.get("skeleton"), dict):
        return None
    return {"skeleton": first.get("skeleton")}


def _sample_video_frames(
    video_path_abs: str,
    *,
    out_dir: str,
    max_frames: int,
) -> List[str]:
    """
    Sample up to max_frames frames using cv2. Returns saved PNG paths.

    Deterministic: evenly spaced indices across the video duration.
    """
    cap = cv2.VideoCapture(video_path_abs)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            idxs = list(range(max_frames))
        else:
            if max_frames <= 1:
                idxs = [max(0, total // 2)]
            else:
                step = max(1, total // max_frames)
                idxs = [min(total - 1, i * step) for i in range(max_frames)]

        os.makedirs(out_dir, exist_ok=True)
        paths: List[str] = []
        for i, frame_index in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out_path = os.path.join(out_dir, f"frame_{i:02d}.png")
            ok_w = cv2.imwrite(out_path, frame)
            if ok_w:
                paths.append(out_path)
        return paths
    finally:
        cap.release()


async def pick_hero_frame_from_video(
    video_path: str,
    *,
    lock_bundle: Dict[str, Any],
    thresholds: Dict[str, float],
    upload_dir: str,
    max_frames: int = 8,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "index": int,
        "score": float|None,
        "locks": {...},
        "image_path": str,
        "source_video": str,
        "candidates": [...]
      }
    """
    video_abs = _abs_upload_path(upload_dir, video_path)
    if not (isinstance(video_abs, str) and video_abs and os.path.exists(video_abs)):
        return {}

    out_dir = os.path.join(upload_dir, "artifacts", "film2", f"hero_frames_{int(time.time())}")
    frame_paths = _sample_video_frames(video_abs, out_dir=out_dir, max_frames=int(max_frames))
    if not frame_paths:
        return {}

    face_refs = _derive_face_refs(lock_bundle)
    region_refs = _derive_region_refs(lock_bundle)
    scene_ref = _extract_scene_ref(lock_bundle)
    style_ref = _derive_style_ref(lock_bundle)
    pose_ref = _derive_pose_ref(lock_bundle)

    face_min = float(thresholds.get("face_min", 0.0)) if isinstance(thresholds, dict) else 0.0
    region_shape_min_t = float(thresholds.get("region_shape_min", 0.0)) if isinstance(thresholds, dict) else 0.0
    region_texture_min_t = float(thresholds.get("region_texture_min", 0.0)) if isinstance(thresholds, dict) else 0.0
    scene_min_t = float(thresholds.get("scene_min", 0.0)) if isinstance(thresholds, dict) else 0.0

    best: Optional[Tuple[int, float, Dict[str, Any]]] = None
    candidates: List[Dict[str, Any]] = []

    for idx, img_path in enumerate(frame_paths):
        face_lock: Optional[float] = None
        region_shape_min: Optional[float] = None
        region_shape_mean: Optional[float] = None
        region_texture_min: Optional[float] = None
        region_texture_mean: Optional[float] = None
        region_color_mean: Optional[float] = None
        region_clip_mean: Optional[float] = None
        scene_score: Optional[float] = None
        style_score: Optional[float] = None
        pose_score: Optional[float] = None

        if face_refs:
            face_detail = await compute_multiface_identity_scores(img_path, face_refs, max_detected_faces=16)
            fl = face_detail.get("aggregate") if isinstance(face_detail, dict) else None
            if isinstance(fl, (int, float)):
                face_lock = float(fl)

        if isinstance(region_refs, dict) and region_refs:
            shape_vals: List[float] = []
            texture_vals: List[float] = []
            color_vals: List[float] = []
            clip_vals: List[float] = []
            for _rid, rdat in region_refs.items():
                if not isinstance(rdat, dict):
                    continue
                scores = await compute_region_scores(img_path, rdat)
                sv = scores.get("shape_score") if isinstance(scores, dict) else None
                if isinstance(sv, (int, float)):
                    shape_vals.append(float(sv))
                tv = scores.get("texture_score") if isinstance(scores, dict) else None
                if isinstance(tv, (int, float)):
                    texture_vals.append(float(tv))
                cv = scores.get("color_score") if isinstance(scores, dict) else None
                if isinstance(cv, (int, float)):
                    color_vals.append(float(cv))
                clv = scores.get("clip_lock") if isinstance(scores, dict) else None
                if isinstance(clv, (int, float)):
                    clip_vals.append(float(clv))
            if shape_vals:
                region_shape_min = float(min(shape_vals))
                region_shape_mean = float(sum(shape_vals) / float(len(shape_vals)))
            if texture_vals:
                region_texture_min = float(min(texture_vals))
                region_texture_mean = float(sum(texture_vals) / float(len(texture_vals)))
            if color_vals:
                region_color_mean = float(sum(color_vals) / float(len(color_vals)))
            if clip_vals:
                region_clip_mean = float(sum(clip_vals) / float(len(clip_vals)))

        if isinstance(scene_ref, dict):
            sc = await compute_scene_score(img_path, scene_ref)
            if isinstance(sc, (int, float)):
                scene_score = float(sc)

        if isinstance(style_ref, dict):
            st = await compute_style_similarity(img_path, style_ref)
            if isinstance(st, (int, float)):
                style_score = float(st)

        if isinstance(pose_ref, dict):
            ps = await compute_pose_similarity(img_path, pose_ref)
            if isinstance(ps, (int, float)):
                pose_score = float(ps)

        locks_out: Dict[str, Any] = {
            "face_lock": face_lock,
            "region_shape_min": region_shape_min,
            "region_shape_mean": region_shape_mean,
            "region_texture_min": region_texture_min,
            "region_texture_mean": region_texture_mean,
            "region_color_mean": region_color_mean,
            "region_clip_mean": region_clip_mean,
            "scene_score": scene_score,
            "style_score": style_score,
            "pose_score": pose_score,
        }

        # Threshold enforcement
        if isinstance(face_lock, (int, float)) and face_min > 0.0 and float(face_lock) < face_min:
            candidates.append({"index": idx, "score": None, "locks": locks_out, "image_path": img_path, "discarded": "face_min"})
            continue
        if isinstance(region_shape_min, (int, float)) and region_shape_min_t > 0.0 and float(region_shape_min) < region_shape_min_t:
            candidates.append({"index": idx, "score": None, "locks": locks_out, "image_path": img_path, "discarded": "region_shape_min"})
            continue
        if isinstance(region_texture_min, (int, float)) and region_texture_min_t > 0.0 and float(region_texture_min) < region_texture_min_t:
            candidates.append({"index": idx, "score": None, "locks": locks_out, "image_path": img_path, "discarded": "region_texture_min"})
            continue
        if isinstance(scene_score, (int, float)) and scene_min_t > 0.0 and float(scene_score) < scene_min_t:
            candidates.append({"index": idx, "score": None, "locks": locks_out, "image_path": img_path, "discarded": "scene_min"})
            continue

        weighted: List[Tuple[float, float]] = []
        if isinstance(face_lock, (int, float)):
            weighted.append((0.45, float(face_lock)))
        if isinstance(region_shape_min, (int, float)):
            weighted.append((0.15, float(region_shape_min)))
        if isinstance(region_texture_min, (int, float)):
            weighted.append((0.15, float(region_texture_min)))
        if isinstance(scene_score, (int, float)):
            weighted.append((0.10, float(scene_score)))
        if isinstance(style_score, (int, float)):
            weighted.append((0.075, float(style_score)))
        if isinstance(pose_score, (int, float)):
            weighted.append((0.075, float(pose_score)))
        if not weighted:
            candidates.append({"index": idx, "score": None, "locks": locks_out, "image_path": img_path, "discarded": "no_scores"})
            continue

        score = sum(w * v for w, v in weighted) / sum(w for w, _v in weighted)
        candidates.append({"index": idx, "score": float(score), "locks": locks_out, "image_path": img_path})
        if best is None or score > best[1]:
            best = (idx, float(score), locks_out | {"_frame_path": img_path})

    if best is None:
        return {"index": 0, "score": None, "locks": {}, "image_path": frame_paths[0], "source_video": video_abs}

    best_idx, best_score, locks_with_path = best
    img_path = locks_with_path.pop("_frame_path")
    return {
        "index": int(best_idx),
        "score": float(best_score),
        "locks": {k: v for k, v in locks_with_path.items() if k != "_frame_path"},
        "image_path": img_path,
        "source_video": video_abs,
        "candidates": candidates,
    }




