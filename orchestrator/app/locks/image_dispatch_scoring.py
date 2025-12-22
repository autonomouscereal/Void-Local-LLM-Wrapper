from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..tracing.runtime import trace_event
from .runtime import bundle_to_image_locks
from .image_lock_scoring import (
    compute_multiface_identity_scores,
    compute_pose_similarity,
    compute_region_scores,
    compute_scene_score,
    compute_style_similarity,
)


async def score_image_dispatch_outputs(
    *,
    saved_paths: List[str],
    lock_bundle: Optional[Dict[str, Any]],
    trace_id: Optional[str],
    artifact_group_id: str,
    quality_profile: str,
) -> Dict[str, Any]:
    """
    Compute lock-domain metrics for image.dispatch outputs.

    Returns a dict suitable for result_obj["meta"]["locks"], including:
      - face_lock (strict min across detected/assigned refs)
      - faces (per-entity face lock mins)
      - style_score / pose_score / scene_score
      - regions metrics
      - entity_lock_score (strict min across entity-level region locks)
      - per_image: detailed breakdown keyed by path
    """
    if not saved_paths:
        return {}
    if not isinstance(lock_bundle, dict):
        lock_bundle = {}

    locks_payload = bundle_to_image_locks(lock_bundle) if lock_bundle else {}
    face_refs = locks_payload.get("faces") if isinstance(locks_payload.get("faces"), list) else []
    region_refs = locks_payload.get("regions") if isinstance(locks_payload.get("regions"), dict) else {}
    style_tags = locks_payload.get("style_tags") if isinstance(locks_payload.get("style_tags"), list) else []
    style_palette = locks_payload.get("style_palette") if isinstance(locks_payload.get("style_palette"), dict) else {}
    style_ref = {"style_tags": style_tags, "palette": style_palette} if (style_tags or style_palette) else None
    poses = locks_payload.get("poses") if isinstance(locks_payload.get("poses"), list) else []
    pose_ref = None
    if poses and isinstance(poses[0], dict) and isinstance(poses[0].get("skeleton"), dict):
        pose_ref = {"skeleton": poses[0].get("skeleton")}
    scene_ref = locks_payload.get("scene") if isinstance(locks_payload.get("scene"), dict) else None

    per_image: Dict[str, Any] = {}
    face_scores: List[float] = []
    face_entity_scores: Dict[str, List[float]] = {}
    style_scores: List[float] = []
    pose_scores: List[float] = []
    scene_scores: List[float] = []
    all_region_metrics: Dict[str, Dict[str, Any]] = {}
    entity_lock_candidates: List[float] = []

    for pth in saved_paths:
        if not isinstance(pth, str) or not pth:
            continue
        img_metrics: Dict[str, Any] = {}

        # Face identity lock
        if face_refs:
            face_detail = await compute_multiface_identity_scores(pth, face_refs, max_detected_faces=16)
            img_metrics["faces"] = face_detail
            fl_val = face_detail.get("aggregate") if isinstance(face_detail, dict) else None
            if isinstance(fl_val, (int, float)):
                face_scores.append(float(fl_val))
            by_ent = face_detail.get("by_entity") if isinstance(face_detail, dict) and isinstance(face_detail.get("by_entity"), dict) else {}
            if isinstance(by_ent, dict):
                for eid, info in by_ent.items():
                    if not isinstance(info, dict):
                        continue
                    s = info.get("score")
                    if isinstance(s, (int, float)):
                        face_entity_scores.setdefault(str(eid), []).append(float(s))

        # Style lock
        if isinstance(style_ref, dict):
            st_val = await compute_style_similarity(pth, style_ref)
            if isinstance(st_val, (int, float)):
                img_metrics["style_score"] = float(st_val)
                style_scores.append(float(st_val))

        # Pose lock
        if isinstance(pose_ref, dict):
            ps_val = await compute_pose_similarity(pth, pose_ref)
            if isinstance(ps_val, (int, float)):
                img_metrics["pose_score"] = float(ps_val)
                pose_scores.append(float(ps_val))

        # Scene lock
        if isinstance(scene_ref, dict):
            sc_val = await compute_scene_score(pth, scene_ref)
            if isinstance(sc_val, (int, float)):
                img_metrics["scene_score"] = float(sc_val)
                scene_scores.append(float(sc_val))

        # Region locks
        if isinstance(region_refs, dict) and region_refs:
            region_out: Dict[str, Any] = {}
            for rid, rdat in region_refs.items():
                if not isinstance(rdat, dict):
                    continue
                scores = await compute_region_scores(pth, rdat)
                if isinstance(scores, dict):
                    region_out[str(rid)] = scores
                    # Track global region aggregates for top-level
                    all_region_metrics[str(rid)] = scores
                    # Track entity_lock candidates when present
                    cv = scores.get("clip_lock")
                    tv = scores.get("texture_score") or scores.get("texture_lock")
                    sv = scores.get("shape_score") or scores.get("shape_lock")
                    vals = []
                    for v in (cv, tv, sv):
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
                    if vals:
                        entity_lock_candidates.append(float(min(vals)))
            img_metrics["regions"] = region_out

        per_image[pth] = img_metrics

    out: Dict[str, Any] = {"bundle": lock_bundle, "quality_profile": str(quality_profile or "")}
    if face_scores:
        out["face_lock"] = float(min(face_scores))
    if face_entity_scores:
        out["faces"] = {k: {"face_lock": float(min(v))} for k, v in face_entity_scores.items() if v}
    if style_scores:
        out["style_score"] = float(sum(style_scores) / float(len(style_scores)))
    if pose_scores:
        out["pose_score"] = float(sum(pose_scores) / float(len(pose_scores)))
    if scene_scores:
        out["scene_score"] = float(sum(scene_scores) / float(len(scene_scores)))
    if all_region_metrics:
        out["regions"] = all_region_metrics
    if entity_lock_candidates:
        out["entity_lock_score"] = float(min(entity_lock_candidates))
    out["per_image"] = per_image

    if isinstance(trace_id, str) and trace_id:
        trace_event(
            "image.lock_scores",
            {
                "trace_id": trace_id,
                "tool": "image.dispatch",
                "artifact_group_id": artifact_group_id,
                "quality_profile": quality_profile,
                "lock_bundle": lock_bundle,
                "face_refs": face_refs,
                "region_refs": region_refs,
                "style_ref": style_ref,
                "pose_ref": pose_ref,
                "scene_ref": scene_ref,
                "face_lock": out.get("face_lock"),
                "style_score": out.get("style_score"),
                "pose_score": out.get("pose_score"),
                "scene_score": out.get("scene_score"),
                "entity_lock_score": out.get("entity_lock_score"),
            },
        )

    return out


