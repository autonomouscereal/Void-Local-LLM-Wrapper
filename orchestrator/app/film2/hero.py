from __future__ import annotations

from typing import Any, Dict, List, Optional


def _extract_meta(frame: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(frame, dict):
        return {}
    meta = frame.get("meta")
    if isinstance(meta, dict):
        return meta
    result = frame.get("result")
    if isinstance(result, dict):
        meta_result = result.get("meta")
        if isinstance(meta_result, dict):
            return meta_result
    return {}


def score_frame_for_locks(
    frame: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Optional[float]:
    meta = _extract_meta(frame)
    if not meta:
        return None
    locks = meta.get("locks") if isinstance(meta.get("locks"), dict) else {}
    face_score = locks.get("face_score")
    region_shape_score = locks.get("region_shape_score")
    scene_score = locks.get("scene_score")
    face_min = thresholds.get("face_min") or 0.0
    region_min = thresholds.get("region_shape_min") or 0.0
    if face_score is not None and face_score < face_min:
        return None
    if region_shape_score is not None and region_shape_score < region_min:
        return None
    weights = [
        (0.5, face_score),
        (0.3, region_shape_score),
        (0.2, scene_score),
    ]
    score = 0.0
    total_weight = 0.0
    for weight, value in weights:
        if isinstance(value, (int, float)):
            score += weight * float(value)
            total_weight += weight
    if total_weight == 0.0:
        return None
    return score / total_weight


def choose_hero_frame(
    frames: List[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    best_frame: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    for idx, frame in enumerate(frames or []):
        score = score_frame_for_locks(frame, thresholds)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_frame = {
                "index": idx,
                "score": score,
                "frame": frame,
            }
    return best_frame

