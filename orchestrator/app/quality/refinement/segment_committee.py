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
    """
    Canonical segment QA + committee review with an optional patch pass.

    - Builds per-segment QA using quality.metrics.tool_results.compute_domain_qa.
    - Calls quality.decisions.committee.postrun_committee_decide for a patch_plan decision.
    - Filters/enriches patch_plan using quality.segments helpers and executes via apply_patch_plan.

    Returns: (updated tool_results list, committee_outcome dict, patch_results list).
    """
    if not segment_results:
        outcome = {"action": "go", "rationale": "no_segment_results", "patch_plan": []}
        return segment_results, outcome, []

    def _emit(kind: str, payload: Dict[str, Any]) -> None:
        """
        Emit a trace event for this trace_id into the unified checkpoint stream.
        This avoids callback injection and keeps segment QA logging consistent with the orchestrator.
        """
        if not state_dir:
            return
        if not isinstance(payload, dict):
            return
        checkpoints_append_event(state_dir, trace_id, str(kind or "event"), payload)

    def _emit_record(record: Dict[str, Any]) -> None:
        """
        Emit a pre-shaped record dict that includes its own `kind`/`event` discriminator.
        """
        if not state_dir:
            return
        if not isinstance(record, dict):
            return
        kind = str(record.get("kind") or record.get("event") or "event")
        checkpoints_append_event(state_dir, trace_id, kind, record)

    thresholds = _lock_quality_thresholds(quality_profile)
    attempt = 0
    current_results = list(segment_results or [])
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
            parent_cid = None
            meta_part = result_obj.get("meta")
            if isinstance(meta_part, dict) and isinstance(meta_part.get("cid"), str):
                parent_cid = meta_part.get("cid")
            # Back-compat / robustness: some executors echo cid in the tool envelope args, not result.meta.
            if parent_cid is None:
                args_part = tr.get("args")
                if isinstance(args_part, dict) and isinstance(args_part.get("cid"), str) and args_part.get("cid"):
                    parent_cid = args_part.get("cid")
            segs = build_segments_for_tool(tool_name, trace_id=trace_id, cid=parent_cid, result=result_obj)
            for seg in (segs or []):
                if not isinstance(seg, dict):
                    continue
                seg_result = seg.get("result") if isinstance(seg.get("result"), dict) else {}
                single_tool_results: List[Dict[str, Any]] = []
                if seg_result:
                    single_tool_results.append({"name": seg.get("tool"), "result": seg_result})
                if single_tool_results:
                    seg_domain_qa = assets_compute_domain_qa(single_tool_results)
                    domain_name = str(seg.get("domain") or "").strip().lower()
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
                    seg_meta_local = seg.get("meta") if isinstance(seg.get("meta"), dict) else {}
                    ls_val = seg_meta_local.get("lipsync_score")
                    if isinstance(ls_val, (int, float)):
                        scores = dict(scores or {})
                        scores["lipsync"] = float(ls_val)
                    sharp_val = seg_meta_local.get("sharpness")
                    if isinstance(sharp_val, (int, float)):
                        scores = dict(scores or {})
                        scores["sharpness"] = float(sharp_val)
                    qa_block = seg.setdefault("qa", {})
                    if isinstance(qa_block, dict):
                        qa_block["scores"] = scores
                segments_for_committee.append(seg)

        for seg in segments_for_committee:
            qa_block = seg.get("qa") if isinstance(seg.get("qa"), dict) else {}
            scores = qa_block.get("scores") if isinstance(qa_block.get("scores"), dict) else {}
            segments_summary.append({"id": seg.get("id"), "domain": seg.get("domain"), "qa": scores, "locks": seg.get("locks") if isinstance(seg.get("locks"), dict) else None})
            seg_meta = seg.get("meta") if isinstance(seg.get("meta"), dict) else {}
            profile_val = seg_meta.get("profile") if isinstance(seg_meta.get("profile"), str) else None
            locks_val = seg.get("locks") if isinstance(seg.get("locks"), dict) else None
            cid_val = seg.get("cid")
            _emit(
                "qa.segment.qa",
                {"trace_id": trace_id, "tool": tool_name, "segment_id": seg.get("id"), "domain": seg.get("domain"), "profile": profile_val, "attempt": attempt, "cid": cid_val, "locks": locks_val, "qa_scores": scores},
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
        texture_val = images_domain.get("region_texture_min")
        if texture_val is None:
            texture_val = images_domain.get("region_texture_mean")
        if texture_val is not None and texture_val < thresholds["region_texture_min"]:
            violations["images.region_texture"] = texture_val
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
        _emit("qa.metrics.segment", {"trace_id": trace_id, "tool": tool_name, "phase": "pre", "attempt": attempt, "metrics": qa_metrics_pre})
        _emit_record({"kind": "qa", "stage": "segment_pre", "data": qa_metrics_pre})

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

        _emit(
            "committee.decision.segment",
            {
                "trace_id": trace_id,
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
                existing_pairs.add((st.get("tool"), st.get("segment_id")))
            for seg in segments_for_committee:
                if not isinstance(seg, dict):
                    continue
                domain_name = str(seg.get("domain") or "").strip().lower()
                if domain_name not in ("video", "film2"):
                    continue
                seg_id = seg.get("id")
                if not isinstance(seg_id, str) or not seg_id:
                    continue
                if "_clip_" not in seg_id:
                    continue
                qa_block = seg.get("qa") if isinstance(seg.get("qa"), dict) else {}
                scores = qa_block.get("scores") if isinstance(qa_block.get("scores"), dict) else {}
                refine_mode_auto: Optional[str] = None
                ls_val = scores.get("lipsync")
                if isinstance(ls_val, (int, float)) and ls_val < 0.5:
                    refine_mode_auto = "fix_lipsync"
                face_val = scores.get("face_lock")
                if refine_mode_auto is None and isinstance(face_val, (int, float)) and face_val < thresholds.get("face_min", 0.0):
                    refine_mode_auto = "fix_faces"
                temporal_val = scores.get("frame_lpips_mean") or scores.get("temporal_stability")
                if refine_mode_auto is None and isinstance(temporal_val, (int, float)) and temporal_val < 0.5:
                    refine_mode_auto = "stabilize_motion"
                sharpness_val = scores.get("sharpness")
                if refine_mode_auto is None and isinstance(sharpness_val, (int, float)) and sharpness_val < 50.0:
                    refine_mode_auto = "improve_quality"
                if refine_mode_auto is None:
                    continue
                key = ("video.refine.clip", seg_id)
                if key in existing_pairs:
                    continue
                auto_patch_steps.append({"tool": "video.refine.clip", "segment_id": seg_id, "args": {"refine_mode": refine_mode_auto}})
            if auto_patch_steps:
                committee_outcome.setdefault("patch_plan", [])
                if isinstance(committee_outcome.get("patch_plan"), list):
                    committee_outcome["patch_plan"].extend(auto_patch_steps)
                seg_id_to_mode: Dict[str, str] = {}
                for step in auto_patch_steps:
                    sid = step.get("segment_id")
                    args_used = step.get("args") if isinstance(step.get("args"), dict) else {}
                    mode_val = args_used.get("refine_mode")
                    if isinstance(sid, str) and sid and isinstance(mode_val, str) and mode_val:
                        seg_id_to_mode[sid] = mode_val
                _emit("film2.clip_refine_plan", {"trace_id": trace_id, "tool": tool_name, "segment_ids": list(seg_id_to_mode.keys()), "refine_modes": seg_id_to_mode})

        if action != "revise":
            break

        raw_patch_plan = committee_outcome.get("patch_plan") or []
        segment_filtered = filter_patch_plan(raw_patch_plan, segments_for_committee)
        filtered_patch_plan: List[Dict[str, Any]] = []
        for step in segment_filtered:
            step_tool = (step.get("tool") or "").strip()
            if not step_tool:
                continue
            filtered_patch_plan.append(step)
        committee_outcome["patch_plan"] = filtered_patch_plan
        if not filtered_patch_plan:
            break

        loop_guard_tools: List[str] = []
        for step in filtered_patch_plan:
            st_name = (step.get("tool") or "").strip()
            if st_name and failure_counts.get(st_name, 0) >= 2:
                loop_guard_tools.append(st_name)
        if loop_guard_tools:
            _emit("committee.loop_guard", {"trace_id": trace_id, "tool": tool_name, "attempt": attempt, "tools": loop_guard_tools})
            committee_outcome["action"] = "fail"
            committee_outcome.setdefault("loop_guard", {"tools": loop_guard_tools})
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
            tname = str((pr or {}).get("name") or "tool")
            result_obj = (pr or {}).get("result") if isinstance((pr or {}).get("result"), dict) else {}
            err_obj = (pr or {}).get("error") or (result_obj.get("error") if isinstance(result_obj, dict) else None)
            if isinstance(err_obj, (str, dict)):
                code = (err_obj.get("code") if isinstance(err_obj, dict) else str(err_obj))
                status = (err_obj.get("status") if isinstance(err_obj, dict) else None)
                message = (err_obj.get("message") if isinstance(err_obj, dict) else "")
                _emit("tool.run.error", {"trace_id": trace_id, "tool": tname, "code": (code or ""), "status": status, "message": (message or ""), "attempt": attempt})
                failure_counts[tname] = int(failure_counts.get(tname, 0) or 0) + 1
            else:
                urls_local = assets_collect_urls([pr], absolutize_url)
                first_url = (urls_local[0] if isinstance(urls_local, list) and urls_local else None)
                _emit("tool.run.after", {"trace_id": trace_id, "tool": tname, "url": first_url, "urls_count": len(urls_local or []), "attempt": attempt})

        if filtered_patch_plan:
            _emit("committee.revision.segment", {"trace_id": trace_id, "tool": tool_name, "steps": len(filtered_patch_plan), "attempt": attempt})
        if patch_results:
            cumulative_patch_results.extend(patch_results)
            current_results = (current_results or []) + patch_results
        attempt += 1
        _emit("segment.refine.iteration", {"trace_id": trace_id, "tool": tool_name, "attempt": attempt})
        if attempt > max_refine_passes:
            break

    if cumulative_patch_results:
        counts_post = {"images": int(assets_count_images(current_results)), "videos": int(assets_count_video(current_results)), "audio": int(assets_count_audio(current_results))}
        domain_post = assets_compute_domain_qa(current_results)
        qa_metrics_post = {"counts": counts_post, "domain": domain_post}
        _emit("qa.metrics.segment", {"trace_id": trace_id, "tool": tool_name, "phase": "post", "metrics": qa_metrics_post})

    return current_results, committee_outcome, cumulative_patch_results


