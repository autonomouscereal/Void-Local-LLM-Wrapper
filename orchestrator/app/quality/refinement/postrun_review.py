from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from ...json_parser import JSONParser
from ...state.checkpoints import append_event as checkpoints_append_event
from ...locks.runtime import quality_thresholds as _lock_quality_thresholds
from ..decisions.committee import postrun_committee_decide
from ..metrics.tool_results import (
    compute_domain_qa as assets_compute_domain_qa,
    count_audio as assets_count_audio,
    count_images as assets_count_images,
    count_video as assets_count_video,
)


def compute_postrun_qa_metrics(tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Shared postrun QA metrics computation (used before/after a revision pass).
    """
    img_count = assets_count_images(tool_results)
    vid_count = assets_count_video(tool_results)
    aud_count = assets_count_audio(tool_results)
    try:
        img_i = int(img_count)
    except Exception:
        img_i = 0
    try:
        vid_i = int(vid_count)
    except Exception:
        vid_i = 0
    try:
        aud_i = int(aud_count)
    except Exception:
        aud_i = 0
    counts_summary = {"images": img_i, "videos": vid_i, "audio": aud_i}
    domain_qa = assets_compute_domain_qa(tool_results)
    # Best-effort thresholds + violations (so committee can reason about "what failed" deterministically).
    # Profile selection: first tool result that declares a quality_profile wins.
    profile_name = "standard"
    for tr in tool_results or []:
        if not isinstance(tr, dict):
            continue
        res = tr.get("result") if isinstance(tr.get("result"), dict) else {}
        meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
        p = meta.get("quality_profile")
        if isinstance(p, str) and p.strip():
            profile_name = p.strip().lower()
            break
        locks_block = meta.get("locks") if isinstance(meta.get("locks"), dict) else {}
        p2 = locks_block.get("quality_profile")
        if isinstance(p2, str) and p2.strip():
            profile_name = p2.strip().lower()
            break

    thresholds: Dict[str, float] = {}
    try:
        thresholds = _lock_quality_thresholds(profile_name)
    except Exception:
        thresholds = _lock_quality_thresholds("standard")

    violations: Dict[str, float] = {}
    images_domain = domain_qa.get("images") if isinstance(domain_qa.get("images"), dict) else {}
    audio_domain = domain_qa.get("audio") if isinstance(domain_qa.get("audio"), dict) else {}
    # Images
    face_min = float(thresholds.get("face_min", 0.80))
    if images_domain.get("face_lock") is not None and isinstance(images_domain.get("face_lock"), (int, float)) and float(images_domain.get("face_lock")) < face_min:
        violations["images.face_lock"] = float(images_domain.get("face_lock"))
    region_shape_min = float(thresholds.get("region_shape_min", 0.70))
    if images_domain.get("region_shape_min") is not None and isinstance(images_domain.get("region_shape_min"), (int, float)) and float(images_domain.get("region_shape_min")) < region_shape_min:
        violations["images.region_shape_min"] = float(images_domain.get("region_shape_min"))
    region_tex_min = float(thresholds.get("region_texture_min", 0.60))
    texture_val = images_domain.get("region_texture_min")
    if texture_val is None:
        texture_val = images_domain.get("region_texture_mean")
    if texture_val is not None and isinstance(texture_val, (int, float)) and float(texture_val) < region_tex_min:
        violations["images.region_texture"] = float(texture_val)
    scene_min = float(thresholds.get("scene_min", 0.60))
    if images_domain.get("scene_lock") is not None and isinstance(images_domain.get("scene_lock"), (int, float)) and float(images_domain.get("scene_lock")) < scene_min:
        violations["images.scene_lock"] = float(images_domain.get("scene_lock"))
    # Audio
    voice_min = float(thresholds.get("voice_min", 0.70))
    if audio_domain.get("voice_lock") is not None and isinstance(audio_domain.get("voice_lock"), (int, float)) and float(audio_domain.get("voice_lock")) < voice_min:
        violations["audio.voice_lock"] = float(audio_domain.get("voice_lock"))
    tempo_min = float(thresholds.get("tempo_min", 0.65))
    if audio_domain.get("tempo_lock") is not None and isinstance(audio_domain.get("tempo_lock"), (int, float)) and float(audio_domain.get("tempo_lock")) < tempo_min:
        violations["audio.tempo_lock"] = float(audio_domain.get("tempo_lock"))
    key_min = float(thresholds.get("key_min", 0.65))
    if audio_domain.get("key_lock") is not None and isinstance(audio_domain.get("key_lock"), (int, float)) and float(audio_domain.get("key_lock")) < key_min:
        violations["audio.key_lock"] = float(audio_domain.get("key_lock"))
    stem_min = float(thresholds.get("stem_balance_min", 0.60))
    if audio_domain.get("stem_balance_lock") is not None and isinstance(audio_domain.get("stem_balance_lock"), (int, float)) and float(audio_domain.get("stem_balance_lock")) < stem_min:
        violations["audio.stem_balance_lock"] = float(audio_domain.get("stem_balance_lock"))
    lyrics_min = float(thresholds.get("lyrics_min", 0.55))
    if audio_domain.get("lyrics_lock") is not None and isinstance(audio_domain.get("lyrics_lock"), (int, float)) and float(audio_domain.get("lyrics_lock")) < lyrics_min:
        violations["audio.lyrics_lock"] = float(audio_domain.get("lyrics_lock"))

    return {
        "counts": counts_summary,
        "domain": domain_qa,
        "profile": profile_name,
        "thresholds": thresholds,
        "threshold_violations": violations,
    }


async def decide_postrun_committee(
    *,
    trace_id: str,
    user_text: str,
    tool_results: List[Dict[str, Any]],
    qa_metrics: Dict[str, Any],
    mode: str,
    state_dir: str = "",
) -> Dict[str, Any]:
    """
    Committee decision wrapper that also emits canonical trace events.
    """
    if state_dir:
        checkpoints_append_event(state_dir, trace_id, "qa.metrics", {"trace_id": trace_id, "tool": "postrun", "metrics": qa_metrics})
        checkpoints_append_event(state_dir, trace_id, "committee.postrun.review", {"trace_id": trace_id, "summary": qa_metrics})
    outcome = await postrun_committee_decide(
        trace_id=trace_id,
        user_text=user_text,
        tool_results=tool_results,
        qa_metrics=qa_metrics,
        mode=mode,
        state_dir=state_dir,
    )
    action = str((outcome.get("action") or "go")).strip().lower() if isinstance(outcome, dict) else "go"
    rationale = str(outcome.get("rationale") or "") if isinstance(outcome, dict) else ""
    if state_dir:
        checkpoints_append_event(state_dir, trace_id, "committee.decision", {"trace_id": trace_id, "action": action, "rationale": rationale})
        checkpoints_append_event(state_dir, trace_id, "committee.decision.final", {"trace_id": trace_id, "action": action})
    return outcome if isinstance(outcome, dict) else {"action": "go", "patch_plan": [], "rationale": "committee_outcome_not_dict"}


def normalize_patch_plan_to_tool_calls(
    patch_plan: Any,
    *,
    planner_visible_tools: Sequence[str],
    allowed_tools: Set[str],
) -> List[Dict[str, Any]]:
    """
    Normalize committee patch_plan into internal tool_calls shape:
      [{"tool_name": <tool>, "arguments": <dict>}]

    Enforces:
    - Accepts any tool name (patch_plan tools can be internal refinement tools like locks.*).
    - Tool validity is checked downstream by execute_tool_call/gateway_execute.
    - Args may be dict, JSON string, None, or arbitrary; unknown types are preserved under _raw.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(patch_plan, list):
        return out
    parser = JSONParser()
    for st in patch_plan:
        if not isinstance(st, dict):
            continue
        # Use tool_name (consistent with planner schema), fallback to "tool" for backward compatibility
        tl = (st.get("tool_name") or st.get("tool") or "").strip() if isinstance(st.get("tool_name") or st.get("tool"), str) else ""
        if not tl:
            continue
        # Accept any tool - validity will be checked by execute_tool_call/gateway_execute
        # Patch plan tools can be internal refinement tools (locks.*) that aren't in the catalog
        args_raw = st.get("args") if ("args" in st) else st.get("arguments")
        args_st: Dict[str, Any]
        if isinstance(args_raw, dict):
            args_st = dict(args_raw)
        elif isinstance(args_raw, str):
            parsed = parser.parse(args_raw or "", dict)
            args_st = dict(parsed) if isinstance(parsed, dict) else {"_raw": args_raw}
        elif args_raw is None:
            args_st = {}
        else:
            args_st = {"_raw": args_raw}
        
        # Normalize argument formats for lock tools to match what execute_tool_call expects
        if tl == "locks.update_audio_modes":
            # execute_tool_call expects "update" (singular), but committee may generate "updates" (plural)
            if "updates" in args_st and "update" not in args_st:
                args_st["update"] = args_st.pop("updates")
        
        out.append({"tool_name": tl, "arguments": args_st})
    return out


