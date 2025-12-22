from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...tracing.runtime import trace_event
from ...locks.runtime import bundle_to_image_locks
from ...locks.image_lock_scoring import compute_multiface_identity_scores


def _face_min_from_quality_profile(quality_profile: str, presets: Dict[str, Any]) -> float:
    prof = str(quality_profile or "").strip().lower() or "standard"
    preset = presets.get(prof) if isinstance(presets, dict) else None
    if not isinstance(preset, dict):
        preset = presets.get("standard") if isinstance(presets, dict) else None
    face_min_raw = (preset or {}).get("face_min", 0.0)
    try:
        face_min = float(face_min_raw)
    except Exception:
        face_min = 0.0
    return float(max(0.0, min(1.0, face_min)))


async def compute_refine_clip_lock_qa(
    *,
    cleaned_frame_paths: List[str],
    lock_bundle: Optional[Dict[str, Any]],
    lock_quality_presets: Dict[str, Any],
    quality_profile: str,
    trace_id: Optional[str],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute identity/lock QA for a refined clip from its cleaned frames.

    Important:
    - Returns embeddings (lock bundle + face refs) so they can be passed downstream and distilled.
    - Also returns explicit context (entity ids/roles, thresholds, sources) so embeddings are never "random vectors".
    - Never raises; returns ok=False on failure so the main flow continues.
    """
    if not cleaned_frame_paths:
        out = {"ok": False, "error": {"code": "no_frames", "message": "no cleaned frames to score"}}
        if isinstance(trace_id, str) and trace_id:
            trace_event("video.refine.clip.qa", {"trace_id": trace_id, "ok": False, **(context or {}), "error": out["error"]})
        return out

    bundle = lock_bundle if isinstance(lock_bundle, dict) else {}
    locks_payload = bundle_to_image_locks(bundle) if bundle else {}
    face_refs = locks_payload.get("faces") if isinstance(locks_payload.get("faces"), list) else []
    if not face_refs:
        out = {
            "ok": True,
            "note": "no face refs in lock bundle; identity scoring skipped",
            "locks": {"bundle": bundle, "quality_profile": quality_profile, "face_refs": face_refs},
        }
        if isinstance(trace_id, str) and trace_id:
            trace_event("video.refine.clip.qa", {"trace_id": trace_id, "ok": True, **(context or {}), "skipped": "no_face_refs"})
        return out

    scores: List[float] = []
    by_entity: Dict[str, List[float]] = {}
    used_entities: List[Dict[str, Any]] = []
    for f in face_refs:
        if not isinstance(f, dict):
            continue
        eid = f.get("entity_id")
        if not isinstance(eid, str) or not eid:
            continue
        used_entities.append({"entity_id": eid, "role": f.get("role"), "priority": f.get("priority")})

    for fp in cleaned_frame_paths:
        if not isinstance(fp, str) or not fp:
            continue
        try:
            detail = await compute_multiface_identity_scores(fp, face_refs, max_detected_faces=16)
        except Exception as ex:
            # Best-effort QA: do not break refine flow if a single frame score fails.
            if isinstance(trace_id, str) and trace_id:
                trace_event(
                    "video.refine.clip.qa_error",
                    {
                        "trace_id": trace_id,
                        **(context or {}),
                        "frame_path": fp,
                        "error": str(ex),
                    },
                )
            continue
        agg = detail.get("aggregate") if isinstance(detail, dict) else None
        if isinstance(agg, (int, float)):
            scores.append(float(agg))
        be = detail.get("by_entity") if isinstance(detail, dict) and isinstance(detail.get("by_entity"), dict) else {}
        if isinstance(be, dict):
            for eid, info in be.items():
                if not isinstance(info, dict):
                    continue
                sc = info.get("score")
                if isinstance(sc, (int, float)):
                    by_entity.setdefault(str(eid), []).append(float(sc))

    face_min = _face_min_from_quality_profile(quality_profile, lock_quality_presets)
    identity_status = "unknown"
    id_min = None
    id_mean = None
    if scores:
        id_min = float(min(scores))
        id_mean = float(sum(scores) / float(len(scores)))
        weak_margin = 0.1 * float(face_min)
        if id_min >= face_min and id_mean >= face_min:
            identity_status = "ok"
        elif id_min >= max(0.0, face_min - weak_margin) and id_mean >= max(0.0, face_min - weak_margin):
            identity_status = "weak"
        else:
            identity_status = "fail"

    out = {
        "ok": True,
        "identity": {
            "status": identity_status,
            "face_min_threshold": face_min,
            "min": id_min,
            "mean": id_mean,
            "frames_scored": int(len(scores)),
            "by_entity": {k: {"min": float(min(v)), "mean": float(sum(v) / float(len(v))), "n": len(v)} for k, v in by_entity.items() if v},
            "entities": used_entities,
        },
        "locks": {"bundle": bundle, "quality_profile": quality_profile, "face_refs": face_refs},
    }
    if isinstance(trace_id, str) and trace_id:
        trace_event(
            "video.refine.clip.qa",
            {
                "trace_id": trace_id,
                "ok": True,
                **(context or {}),
                "identity_status": identity_status,
                "face_min_threshold": face_min,
                "identity_min": id_min,
                "identity_mean": id_mean,
                "frames_scored": int(len(scores)),
                "entities": used_entities,
                "lock_bundle": bundle,
                "face_refs": face_refs,
            },
        )
    return out


