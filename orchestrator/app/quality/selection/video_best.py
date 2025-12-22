from __future__ import annotations

from typing import Any, Dict, Optional

from ...tracing.runtime import trace_event
from ...locks.film2_hero_frame_video_locks import pick_hero_frame_from_video


async def choose_best_video_pair(
    *,
    base_path: str,
    processed_path: str,
    base_temporal_stability: Optional[float],
    processed_temporal_stability: Optional[float],
    lock_bundle: Dict[str, Any],
    thresholds_lock: Dict[str, float],
    upload_dir: str,
    trace_id: Optional[str],
    event_kind: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Canonical 'best' chooser for two video candidates that factors:
      - locks (hero frame lock score) AND
      - temporal QA (temporal_stability)

    Returns:
      {
        "best_path": str,
        "best_reason": str,
        "scores": {"base": {...}, "processed": {...}},
        "heroes": {"base": hero_record|None, "processed": hero_record|None},
      }
    """
    base = base_path if isinstance(base_path, str) else ""
    proc = processed_path if isinstance(processed_path, str) else ""
    if not base and not proc:
        return {"best_path": "", "best_reason": "missing_candidates", "scores": {}, "heroes": {}}
    if not proc:
        return {"best_path": base, "best_reason": "base_only", "scores": {}, "heroes": {}}
    if not base:
        return {"best_path": proc, "best_reason": "processed_only", "scores": {}, "heroes": {}}

    # Hero lock scoring (locks system) for each candidate
    base_hero = await pick_hero_frame_from_video(
        base,
        lock_bundle=lock_bundle if isinstance(lock_bundle, dict) else {},
        thresholds=thresholds_lock if isinstance(thresholds_lock, dict) else {},
        upload_dir=upload_dir,
    )
    proc_hero = await pick_hero_frame_from_video(
        proc,
        lock_bundle=lock_bundle if isinstance(lock_bundle, dict) else {},
        thresholds=thresholds_lock if isinstance(thresholds_lock, dict) else {},
        upload_dir=upload_dir,
    )
    base_hs = float(base_hero.get("score")) if isinstance(base_hero, dict) and isinstance(base_hero.get("score"), (int, float)) else None
    proc_hs = float(proc_hero.get("score")) if isinstance(proc_hero, dict) and isinstance(proc_hero.get("score"), (int, float)) else None

    b_stab = float(base_temporal_stability or 0.0) if base_temporal_stability is not None else 0.0
    p_stab = float(processed_temporal_stability or 0.0) if processed_temporal_stability is not None else 0.0
    b_h = float(base_hs or 0.0) if base_hs is not None else 0.0
    p_h = float(proc_hs or 0.0) if proc_hs is not None else 0.0

    base_score = 0.55 * b_h + 0.45 * b_stab
    proc_score = 0.55 * p_h + 0.45 * p_stab

    scores = {
        "base": {"temporal_stability": base_temporal_stability, "hero_lock_score": base_hs, "score": float(base_score)},
        "processed": {"temporal_stability": processed_temporal_stability, "hero_lock_score": proc_hs, "score": float(proc_score)},
    }

    if proc_score + 0.01 >= base_score:
        best_path = proc
        best_reason = "processed_best"
    else:
        best_path = base
        best_reason = "base_best"

    if isinstance(trace_id, str) and trace_id:
        trace_event(
            event_kind,
            {
                "trace_id": trace_id,
                "lock_bundle": lock_bundle,
                "thresholds_lock": thresholds_lock,
                "base_path": base,
                "processed_path": proc,
                "base_score": float(base_score),
                "processed_score": float(proc_score),
                "base_temporal": base_temporal_stability,
                "processed_temporal": processed_temporal_stability,
                "base_hero": base_hs,
                "processed_hero": proc_hs,
                "best_path": best_path,
                "best_reason": best_reason,
                **(context if isinstance(context, dict) else {}),
            },
        )

    return {
        "best_path": best_path,
        "best_reason": best_reason,
        "scores": scores,
        "heroes": {"base": base_hero if isinstance(base_hero, dict) else None, "processed": proc_hero if isinstance(proc_hero, dict) else None},
    }


