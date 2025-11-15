from __future__ import annotations

from typing import Any, Dict, List


def build_delta_plan(scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collapse primitive scores into an accept/decline decision with a simple patch plan.

    This function now understands per-segment QA passed in under scores["segments"]
    and will emit segment-aware patch steps with segment_id when segments fall
    below hard-coded thresholds. It remains deliberately simple and deterministic
    so that the wrapper can be swapped out for an LLM-based committee later.
    """
    segments: List[Dict[str, Any]] = []
    raw_segments = scores.get("segments")
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if isinstance(item, dict):
                segments.append(item)
    any_violation = False
    patch_plan: List[Dict[str, Any]] = []
    segment_threshold = 0.8
    for seg in segments:
        seg_id = seg.get("id")
        qa_block = seg.get("qa") if isinstance(seg.get("qa"), dict) else {}
        scores_block = qa_block if isinstance(qa_block, dict) else {}
        worst_score = None
        for val in scores_block.values():
            if isinstance(val, (int, float)):
                v = float(val)
                if worst_score is None or v < worst_score:
                    worst_score = v
        if isinstance(seg_id, str) and seg_id and isinstance(worst_score, float) and worst_score < segment_threshold:
            any_violation = True
            domain = seg.get("domain")
            tool_name = None
            if domain == "image":
                tool_name = "image.refine.segment"
            elif domain in ("video", "film2"):
                tool_name = "video.refine.clip"
            elif domain in ("music", "audio"):
                tool_name = "music.refine.window"
            elif domain == "tts":
                tool_name = "tts.refine.segment"
            if tool_name is not None:
                patch_plan.append(
                    {
                        "tool": tool_name,
                        "segment_id": seg_id,
                        "args": {},
                    }
                )
    if not any_violation:
        return {
            "accept": True,
            "reasons": [],
            "patch_plan": [],
            "targets": {},
        }
    return {
        "accept": False,
        "reasons": ["segment_qa_below_threshold"],
        "patch_plan": patch_plan,
        "targets": {"segment_min": segment_threshold},
    }


