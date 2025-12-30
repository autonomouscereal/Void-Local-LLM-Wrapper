from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..segments import (
    build_segments_for_tool,
    filter_patch_plan,
    apply_patch_plan,
    enrich_patch_plan_for_image_segments,
    enrich_patch_plan_for_video_segments,
    enrich_patch_plan_for_music_segments,
    enrich_patch_plan_for_tts_segments,
    enrich_patch_plan_for_sfx_segments,
)
from ..decisions.committee import postrun_committee_decide
from ..metrics.tool_results import count_images as assets_count_images, count_video as assets_count_video, count_audio as assets_count_audio, compute_domain_qa as assets_compute_domain_qa, collect_urls as assets_collect_urls
from ...locks.runtime import quality_thresholds as _lock_quality_thresholds
from ...state.checkpoints import append_event as checkpoints_append_event


async def segment_qa_and_committee(
    trace_id: str,
    user_text: str,
    tool_name: str,
    segment_results: List[Dict[str, Any]],
    mode: str,
    *,
    absolutize_url: Callable[[str], str],
    quality_profile: Optional[str] = None,
    max_refine_passes: int = 1,
    # Optional: legacy call sites may still pass these; they are ignored here
    # (the actual patch execution is provided via tool_runner).
    base_url: Optional[str] = None,
    executor_base_url: Optional[str] = None,
    tool_runner: Optional[Callable[[Dict[str, Any]], Any]] = None,
    state_dir: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    if not isinstance(segment_results, list):
        log.warning(f"segment_committee.segment_qa_and_committee: invalid segment_results type={type(segment_results).__name__!r} tool_name={tool_name!r} trace_id={trace_id!r}")
        segment_results = []
    if not segment_results:
        log.debug(f"segment_committee.segment_qa_and_committee: empty segment_results tool_name={tool_name!r} trace_id={trace_id!r}")
        return [], {"action": "go", "patch_plan": [], "rationale": "no segments to QA"}, []
    """
    Canonical segment QA + committee review with an optional patch pass.

    - Builds per-segment QA using quality.metrics.tool_results.compute_domain_qa.
    - Calls quality.decisions.committee.postrun_committee_decide for a patch_plan decision.
    - Filters/enriches patch_plan using quality.segments helpers and executes via apply_patch_plan.

    Returns: (updated tool_results list, committee_outcome dict, patch_results list).
    """
    current_results = list(segment_results or [])
    if not current_results:
        # Do not early-return silently; emit a structured trace and continue with a default outcome.
        if state_dir:
            checkpoints_append_event(
                state_dir,
                trace_id,
                "qa.segment.empty",
                {"trace_id": trace_id, "tool_name": tool_name, "tool": tool_name, "reason": "no_segment_results", "user_text": user_text},
            )

    # Hard cap: run at most ONE refine pass to prevent loops.
    try:
        max_refine_passes = min(int(max_refine_passes), 1)
    except Exception:
        max_refine_passes = 1

    thresholds = _lock_quality_thresholds(quality_profile)
    attempt = 0
    # current_results already set above; never rely on early return for empty input.
    cumulative_patch_results: List[Dict[str, Any]] = []
    committee_outcome: Dict[str, Any] = {"action": "go", "patch_plan": [], "rationale": ""}
    failure_counts: Dict[str, int] = {}

    while attempt <= max(0, int(max_refine_passes)):
        segments_summary: List[Dict[str, Any]] = []
        segments_for_committee: List[Dict[str, Any]] = []

        for tr in current_results:
            if not isinstance(tr, dict):
                continue
            result_obj = tr.get("result")
            if not isinstance(result_obj, dict):
                continue
            conversation_id = None
            meta_part = result_obj.get("meta")
            if isinstance(meta_part, dict) and isinstance(meta_part.get("conversation_id"), str):
                conversation_id = meta_part.get("conversation_id")
            if conversation_id is None:
                args_part = tr.get("args")
                if isinstance(args_part, dict) and isinstance(args_part.get("conversation_id"), str) and args_part.get("conversation_id"):
                    conversation_id = args_part.get("conversation_id")
            segs = build_segments_for_tool(tool_name, trace_id=trace_id, conversation_id=conversation_id, result=result_obj)
            if not isinstance(segs, list):
                log.warning(f"segment_committee.segment_qa_and_committee: build_segments_for_tool returned non-list type={type(segs).__name__!r} tool_name={tool_name!r} trace_id={trace_id!r}")
                segs = []
            for segment in (segs or []):
                if not isinstance(segment, dict):
                    log.debug(f"segment_committee.segment_qa_and_committee: skipping non-dict segment type={type(segment).__name__!r} tool_name={tool_name!r} trace_id={trace_id!r}")
                    continue
                seg_result = segment.get("result") if isinstance(segment.get("result"), dict) else {}
                single_tool_results: List[Dict[str, Any]] = []
                if seg_result:
                    tool_name_seg = segment.get("tool_name") or segment.get("tool") or ""
                    single_tool_results.append({"tool_name": tool_name_seg, "result": seg_result})
                if single_tool_results:
                    seg_domain_qa = assets_compute_domain_qa(single_tool_results)
                    domain_name = str(segment.get("domain") or "").strip().lower()
                    scores: Dict[str, Any] = {}
                    if domain_name == "image":
                        scores = seg_domain_qa.get("images") or {}
                    elif domain_name in ("music", "tts", "sfx", "audio"):
                        scores = seg_domain_qa.get("audio") or {}
                    elif domain_name == "video":
                        scores = seg_domain_qa.get("videos") or {}
                    elif domain_name == "film2":
                        scores = dict(seg_domain_qa.get("videos") or {})
                        img_scores = seg_domain_qa.get("images") or {}
                        if isinstance(img_scores, dict) and img_scores:
                            scores.update(img_scores)
                    seg_meta_local = segment.get("meta") if isinstance(segment.get("meta"), dict) else {}
                    scores = dict(scores or {})
                    scores["lipsync"] = float(seg_meta_local.get("lipsync_score") or 0.0)
                    scores["sharpness"] = float(seg_meta_local.get("sharpness") or 0.0)
                    qa_block = segment.setdefault("qa", {})
                    if isinstance(qa_block, dict):
                        qa_block["scores"] = scores
                # Ensure trace_id and conversation_id are propagated to segments
                if trace_id and not segment.get("trace_id"):
                    segment["trace_id"] = trace_id
                if conversation_id and not segment.get("conversation_id"):
                    segment["conversation_id"] = conversation_id
                segments_for_committee.append(segment)

        for segment in segments_for_committee:
            qa_block = segment.get("qa") if isinstance(segment.get("qa"), dict) else {}
            scores = qa_block.get("scores") if isinstance(qa_block.get("scores"), dict) else {}
            segments_summary.append({"segment_id": segment.get("segment_id"), "domain": segment.get("domain"), "qa": scores, "locks": segment.get("locks") if isinstance(segment.get("locks"), dict) else None})
            seg_meta = segment.get("meta") if isinstance(segment.get("meta"), dict) else {}
            profile = seg_meta.get("profile") if isinstance(seg_meta.get("profile"), str) else None
            locks = segment.get("locks") if isinstance(segment.get("locks"), dict) else None
            conversation_id = segment.get("conversation_id")
            if state_dir:
                checkpoints_append_event(
                    state_dir,
                    trace_id,
                    "qa.segment.qa",
                    {"trace_id": trace_id, "tool_name": tool_name, "tool": tool_name, "segment_id": segment.get("segment_id"), "domain": segment.get("domain"), "profile": profile, "attempt": attempt, "conversation_id": conversation_id, "locks": locks, "qa_scores": scores},
                )

        counts = {"images": int(assets_count_images(current_results)), "videos": int(assets_count_video(current_results)), "audio": int(assets_count_audio(current_results))}
        domain_qa = assets_compute_domain_qa(current_results)

        violations: Dict[str, float] = {}
        images_domain = domain_qa.get("images") or {}
        audio_domain = domain_qa.get("audio") or {}
        if images_domain.get("face_lock") is not None and images_domain.get("face_lock") < thresholds["face_min"]:
            violations["images.face_lock"] = images_domain.get("face_lock")
        if images_domain.get("region_shape_min") is not None and images_domain.get("region_shape_min") < thresholds["region_shape_min"]:
            violations["images.region_shape_min"] = images_domain.get("region_shape_min")
        texture = images_domain.get("region_texture_min")
        if texture is None:
            texture = images_domain.get("region_texture_mean")
        if texture is not None and texture < thresholds["region_texture_min"]:
            violations["images.region_texture"] = texture
        if images_domain.get("scene_lock") is not None and images_domain.get("scene_lock") < thresholds["scene_min"]:
            violations["images.scene_lock"] = images_domain.get("scene_lock")
        if audio_domain.get("voice_lock") is not None and audio_domain.get("voice_lock") < thresholds["voice_min"]:
            violations["audio.voice_lock"] = audio_domain.get("voice_lock")
        if audio_domain.get("tempo_lock") is not None and audio_domain.get("tempo_lock") < thresholds["tempo_min"]:
            violations["audio.tempo_lock"] = audio_domain.get("tempo_lock")
        if audio_domain.get("key_lock") is not None and audio_domain.get("key_lock") < thresholds["key_min"]:
            violations["audio.key_lock"] = audio_domain.get("key_lock")
        if audio_domain.get("stem_balance_lock") is not None and audio_domain.get("stem_balance_lock") < thresholds["stem_balance_min"]:
            violations["audio.stem_balance_lock"] = audio_domain.get("stem_balance_lock")
        if audio_domain.get("lyrics_lock") is not None and audio_domain.get("lyrics_lock") < thresholds["lyrics_min"]:
            violations["audio.lyrics_lock"] = audio_domain.get("lyrics_lock")

        qa_metrics_pre = {
            "counts": counts,
            "domain": domain_qa,
            "thresholds": thresholds,
            "threshold_violations": violations,
            "segments": segments_summary,
        }
        if state_dir:
            checkpoints_append_event(state_dir, trace_id, "qa.metrics.segment", {"trace_id": trace_id, "tool_name": tool_name, "tool": tool_name, "phase": "pre", "attempt": attempt, "metrics": qa_metrics_pre})
            checkpoints_append_event(state_dir, trace_id, "qa.segment_pre", {"trace_id": trace_id, "tool_name": tool_name, "tool": tool_name, "attempt": attempt, "metrics": qa_metrics_pre})

        committee_outcome = await postrun_committee_decide(
            trace_id=trace_id,
            user_text=user_text,
            tool_results=current_results,
            qa_metrics=qa_metrics_pre,
            mode=mode,
            state_dir=state_dir,
        )
        action = str((committee_outcome.get("action") or "go")).strip().lower()

        if violations and action == "go":
            if attempt < max_refine_passes:
                action = "revise"
                committee_outcome["action"] = "revise"
                rationale = committee_outcome.get("rationale") or ""
                committee_outcome["rationale"] = (rationale + " | auto-revise: lock thresholds unmet").strip()
                committee_outcome.setdefault("threshold_violations", {}).update(violations)
            else:
                committee_outcome["action"] = "fail"
                committee_outcome.setdefault("threshold_violations", {}).update(violations)
                action = "fail"

        if state_dir:
            checkpoints_append_event(
                state_dir,
                trace_id,
                "committee.decision.segment",
                {
                    "trace_id": trace_id,
                    "tool_name": tool_name,
                    "tool": tool_name,
                    "attempt": attempt,
                    "action": action,
                    "rationale": committee_outcome.get("rationale"),
                    "violations": violations,
                    "committee_error": committee_outcome.get("committee_error"),
                },
            )

        if action == "revise":
            auto_patch_steps: List[Dict[str, Any]] = []
            existing_pairs = set()
            for st in committee_outcome.get("patch_plan") or []:
                if not isinstance(st, dict):
                    continue
                existing_pairs.add((st.get("tool_name") or st.get("tool"), st.get("segment_id")))
            for segment in segments_for_committee:
                if not isinstance(segment, dict):
                    continue
                domain_name = str(segment.get("domain") or "").strip().lower()
                if domain_name not in ("video", "film2"):
                    continue
                segment_id = segment.get("segment_id")
                if not isinstance(segment_id, str) or not segment_id:
                    continue
                if "_clip_" not in segment_id:
                    continue
                qa_block = segment.get("qa") if isinstance(segment.get("qa"), dict) else {}
                scores = qa_block.get("scores") if isinstance(qa_block.get("scores"), dict) else {}
                refine_mode_auto: Optional[str] = None
                lipsync = scores.get("lipsync")
                if isinstance(lipsync, (int, float)) and lipsync < 0.5:
                    refine_mode_auto = "fix_lipsync"
                face_lock = scores.get("face_lock")
                if refine_mode_auto is None and isinstance(face_lock, (int, float)) and face_lock < thresholds.get("face_min", 0.0):
                    refine_mode_auto = "fix_faces"
                temporal_stability = scores.get("frame_lpips_mean") or scores.get("temporal_stability")
                if refine_mode_auto is None and isinstance(temporal_stability, (int, float)) and temporal_stability < 0.5:
                    refine_mode_auto = "stabilize_motion"
                sharpness = scores.get("sharpness")
                if refine_mode_auto is None and isinstance(sharpness, (int, float)) and sharpness < 50.0:
                    refine_mode_auto = "improve_quality"
                if refine_mode_auto is None:
                    continue
                key = ("video.refine.clip", segment_id)
                if key in existing_pairs:
                    continue
                auto_patch_steps.append({"tool_name": "video.refine.clip", "tool": "video.refine.clip", "segment_id": segment_id, "args": {"refine_mode": refine_mode_auto}})
            if auto_patch_steps:
                committee_outcome.setdefault("patch_plan", [])
                if isinstance(committee_outcome.get("patch_plan"), list):
                    committee_outcome["patch_plan"].extend(auto_patch_steps)
                seg_id_to_mode: Dict[str, str] = {}
                for step in auto_patch_steps:
                    segment_id = step.get("segment_id")
                    args_used = step.get("args") if isinstance(step.get("args"), dict) else {}
                    refine_mode = args_used.get("refine_mode")
                    if isinstance(segment_id, str) and segment_id and isinstance(refine_mode, str) and refine_mode:
                        seg_id_to_mode[segment_id] = refine_mode
                if state_dir:
                    checkpoints_append_event(state_dir, trace_id, "film2.clip_refine_plan", {"trace_id": trace_id, "tool_name": tool_name, "tool": tool_name, "segment_ids": list(seg_id_to_mode.keys()), "refine_modes": seg_id_to_mode})

        if action != "revise":
            break

        raw_patch_plan = committee_outcome.get("patch_plan") or []
        segment_filtered = filter_patch_plan(raw_patch_plan, segments_for_committee)
        filtered_patch_plan: List[Dict[str, Any]] = []
        for step in segment_filtered:
            step_tool = (step.get("tool_name") or step.get("tool") or "").strip()
            if not step_tool:
                continue
            filtered_patch_plan.append(step)
        committee_outcome["patch_plan"] = filtered_patch_plan
        if not filtered_patch_plan:
            break

        loop_guard_tool_names: List[str] = []
        for step in filtered_patch_plan:
            st_name = (step.get("tool_name") or step.get("tool") or "").strip()
            if st_name and failure_counts.get(st_name, 0) >= 2:
                loop_guard_tool_names.append(st_name)
        if loop_guard_tool_names:
            if state_dir:
                checkpoints_append_event(state_dir, trace_id, "committee.loop_guard", {"trace_id": trace_id, "tool_name": tool_name, "tool": tool_name, "attempt": attempt, "tool_names": loop_guard_tool_names, "tools": loop_guard_tool_names})
            committee_outcome["action"] = "fail"
            committee_outcome.setdefault("loop_guard", {"tool_names": loop_guard_tool_names, "tools": loop_guard_tool_names})
            break

        enriched_patch_plan = enrich_patch_plan_for_image_segments(filtered_patch_plan, segments_for_committee)
        enriched_patch_plan = enrich_patch_plan_for_video_segments(enriched_patch_plan, segments_for_committee)
        enriched_patch_plan = enrich_patch_plan_for_music_segments(enriched_patch_plan, segments_for_committee)
        enriched_patch_plan = enrich_patch_plan_for_tts_segments(enriched_patch_plan, segments_for_committee)
        enriched_patch_plan = enrich_patch_plan_for_sfx_segments(enriched_patch_plan, segments_for_committee)

        if tool_runner is None:
            # Caller did not supply an execution callback (common for SFX where refine passes are 0).
            break

        updated_segments, patch_results = await apply_patch_plan(enriched_patch_plan, segments_for_committee, tool_runner, trace_id, tool_name)

        for pr in patch_results or []:
            # Check tool_name (canonical) first, then name (OpenAI format) as fallback
            patch_tool_name = str((pr or {}).get("tool_name") or (pr or {}).get("name") or "tool")
            result_obj = (pr or {}).get("result") if isinstance((pr or {}).get("result"), dict) else {}
            err_obj = (pr or {}).get("error") or (result_obj.get("error") if isinstance(result_obj, dict) else None)
            if isinstance(err_obj, (str, dict)):
                code = (err_obj.get("code") if isinstance(err_obj, dict) else str(err_obj))
                status = (err_obj.get("status") if isinstance(err_obj, dict) else None)
                message = (err_obj.get("message") if isinstance(err_obj, dict) else "")
                if state_dir:
                    checkpoints_append_event(state_dir, trace_id, "tool.run.error", {"trace_id": trace_id, "tool_name": patch_tool_name, "tool": patch_tool_name, "code": (code or ""), "status": status, "message": (message or ""), "attempt": attempt})
                failure_counts[patch_tool_name] = int(failure_counts.get(patch_tool_name, 0) or 0) + 1
            else:
                urls_local = assets_collect_urls([pr], absolutize_url)
                first_url = (urls_local[0] if isinstance(urls_local, list) and urls_local else None)
                if state_dir:
                    checkpoints_append_event(state_dir, trace_id, "tool.run.after", {"trace_id": trace_id, "tool_name": patch_tool_name, "tool": patch_tool_name, "url": first_url, "urls_count": len(urls_local or []), "attempt": attempt})

        if filtered_patch_plan:
            if state_dir:
                checkpoints_append_event(state_dir, trace_id, "committee.revision.segment", {"trace_id": trace_id, "tool_name": tool_name, "tool": tool_name, "steps": len(filtered_patch_plan), "attempt": attempt})
        if patch_results:
            cumulative_patch_results.extend(patch_results)
            # Rebuild current_results from updated_segments to include the patched segment data
            # Convert updated_segments back to tool_results format for next iteration
            current_results = []
            for seg in updated_segments or []:
                if isinstance(seg, dict):
                    seg_result = seg.get("result") if isinstance(seg.get("result"), dict) else {}
                    if seg_result:
                        seg_tool_name = seg.get("tool_name") or seg.get("tool") or tool_name
                        current_results.append({"tool_name": seg_tool_name, "tool": seg_tool_name, "result": seg_result, "args": seg.get("args") if isinstance(seg.get("args"), dict) else {}})
            # Also append patch_results as tool results for tracking
            current_results.extend(patch_results)
        attempt += 1
        if state_dir:
            checkpoints_append_event(state_dir, trace_id, "segment.refine.iteration", {"trace_id": trace_id, "tool_name": tool_name, "tool": tool_name, "attempt": attempt})
        if attempt > max_refine_passes:
            break

    if cumulative_patch_results:
        counts_post = {"images": int(assets_count_images(current_results)), "videos": int(assets_count_video(current_results)), "audio": int(assets_count_audio(current_results))}
        domain_post = assets_compute_domain_qa(current_results)
        qa_metrics_post = {"counts": counts_post, "domain": domain_post}
        if state_dir:
            checkpoints_append_event(state_dir, trace_id, "qa.metrics.segment", {"trace_id": trace_id, "tool_name": tool_name, "tool": tool_name, "phase": "post", "metrics": qa_metrics_post})

    return current_results, committee_outcome, cumulative_patch_results


