from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ..refinement.segment_committee import segment_qa_and_committee
from ...state.checkpoints import append_event as checkpoints_append_event
from ...tracing.runtime import trace_event

log = logging.getLogger(__name__)


def tool_step_eval_user_text(tool_name: str, tool_args: Any, last_user_text: str) -> str:
    """
    Committee/QA prompt seed that is scoped to a SINGLE tool execution step.

    NOTE: This is intentionally lightweight and deterministic; it should not pull
    in orchestrator/main logging utilities to avoid circular imports.
    """
    a = tool_args if isinstance(tool_args, dict) else {"_raw": tool_args}
    # Primary: tool-run intent (what THIS tool call is trying to accomplish).
    goal_hint = ""
    if isinstance(tool_name, str) and tool_name.startswith("image."):
        p = a.get("prompt") or a.get("text") or ""
        if isinstance(p, str) and p.strip():
            goal_hint = f"Generate/refine image(s) to match tool-intent prompt={p.strip()}."
    elif isinstance(tool_name, str) and tool_name.startswith("video."):
        src = a.get("src") or a.get("video") or a.get("url") or ""
        if isinstance(src, str) and src.strip():
            goal_hint = f"Process video src={src.strip()} per tool intent."
    elif isinstance(tool_name, str) and (tool_name.startswith("music.") or tool_name.startswith("tts.") or tool_name.startswith("sfx.")):
        p = a.get("prompt") or a.get("text") or a.get("lyrics") or ""
        if isinstance(p, str) and p.strip():
            goal_hint = f"Generate/refine audio to match tool-intent prompt/text={p.strip()}."
    elif tool_name == "film2.run":
        syn = a.get("synopsis") or a.get("prompt") or ""
        if isinstance(syn, str) and syn.strip():
            goal_hint = f"Generate film output to match tool-intent synopsis/prompt={syn.strip()}."

    # Full-fidelity debugging: do not truncate or whitelist.
    # json.dumps(... default=str) below ensures non-JSON types don't crash the eval prompt builder.
    args_preview: Dict[str, Any]
    if not isinstance(a, dict):
        args_preview = {"type": type(a).__name__, "value": a}
    else:
        keys = sorted([str(k) for k in a.keys()])
        args_preview = {"keys": keys, "has_raw": bool("_raw" in a), "args": {str(k): v for k, v in a.items()}}

    # Global context (secondary): original user request. This is included so the committee can
    # verify the tool intent is consistent with the larger project, but the PRIMARY evaluation
    # target is the tool intent + tool args + locks/bundles/QA.
    blob = {
        "scope": "tool_step_only",
        "instruction": {
            "primary_goal": "Evaluate ONLY this SINGLE tool run against its own tool intent + tool args + locks/bundles + QA/quality results.",
            "secondary_goal": "Use the original user request ONLY as context to confirm the tool intent is consistent with the larger project (movie/story), not as the primary matching target for this step.",
            "explicit_non_goals": [
                "Do NOT evaluate whether the entire user request has been completed.",
                "Do NOT judge this tool step against the original user request directly if the tool intent is narrower/different.",
                "Do NOT propose unrelated tools.",
                "Only propose revisions/patches if this tool step's own outputs/QA/locks are below thresholds or the step failed.",
            ],
        },
        "global_context": {
            "original_user_request": (last_user_text or ""),
        },
        "tool_name": tool_name,
        "tool": tool_name,
        "tool_intent_primary": goal_hint,
        "tool_args_full": args_preview,
        "locks_and_bundles": {
            # If a lock_bundle is provided, include it (full fidelity). This is what the
            # step must obey for consistency (identity/style/tempo/voice/etc).
            "lock_bundle": a.get("lock_bundle") if isinstance(a, dict) else None,
            "voice_lock_id": a.get("voice_lock_id") if isinstance(a, dict) else None,
            "character_id": a.get("character_id") if isinstance(a, dict) else None,
            "segment_id": a.get("segment_id") if isinstance(a, dict) else None,
        },
    }
    return json.dumps(blob, ensure_ascii=False, default=str)


def _clamp_refine_budget(preset: Dict[str, Any], *, default: int, cap: int = 1) -> int:
    raw = preset.get("max_refine_passes", default) if isinstance(preset, dict) else default
    try:
        return min(int(cap), max(0, int(raw)))
    except Exception:
        return min(int(cap), max(0, int(default)))


def _get_profile_and_preset(
    result_obj: Dict[str, Any],
    quality_presets: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    meta_obj = result_obj.get("meta") if isinstance(result_obj.get("meta"), dict) else {}
    profile_name = meta_obj.get("quality_profile") if isinstance(meta_obj.get("quality_profile"), str) else None
    preset = quality_presets.get((profile_name or "standard").lower(), quality_presets.get("standard", {}))
    return profile_name, preset, meta_obj


async def _segment_qa_for_single_result(
    *,
    trace_id: str,
    tool_name: str,
    tool_args: Any,
    last_user_text: str,
    tool_trace_result: Dict[str, Any],
    effective_mode: str,
    absolutize_url: Callable[[str], str],
    profile_name: Optional[str],
    refine_budget: int,
    state_dir: str,
    tool_runner: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not isinstance(trace_id, str) or not trace_id:
        log.error(f"tool_step_qa._segment_qa_for_single_result: missing trace_id tool_name={tool_name!r} - upstream caller must pass trace_id. Continuing with empty trace_id but this is an error.")
        trace_id = ""  # Continue processing but log the error - NO FALLBACK GENERATION
    if not isinstance(tool_trace_result, dict):
        log.warning(f"tool_step_qa._segment_qa_for_single_result: invalid tool_trace_result type={type(tool_trace_result).__name__!r} tool_name={tool_name!r} trace_id={trace_id!r}")
        return {"segment_committee": {"action": "go", "patch_plan": [], "rationale": f"invalid tool_trace_result type: {type(tool_trace_result).__name__}"}}, []
    try:
        trace_event("qa.segment.start", {"trace_id": trace_id, "tool_name": tool_name})
        extra_results: List[Dict[str, Any]] = []
        _, seg_outcome, seg_patch = await segment_qa_and_committee(
            trace_id=trace_id,
            user_text=tool_step_eval_user_text(tool_name, tool_args, last_user_text),
            tool_name=tool_name,
            segment_results=[tool_trace_result],
            mode=effective_mode,
            absolutize_url=absolutize_url,
            quality_profile=profile_name,
            max_refine_passes=refine_budget,
            tool_runner=(tool_runner if refine_budget > 0 else None),
            state_dir=state_dir,
        )
        meta_patch: Dict[str, Any] = {"segment_committee": seg_outcome}
        if seg_patch:
            meta_patch["segment_patch_results"] = seg_patch
            extra_results.extend(seg_patch)
        trace_event("qa.segment.complete", {"trace_id": trace_id, "tool_name": tool_name, "patch_count": len(seg_patch)})
        return meta_patch, extra_results
    except Exception as ex:
        log.error(f"tool_step_qa._segment_qa_for_single_result.exception trace_id={trace_id!r} tool_name={tool_name!r} ex={ex!r}", exc_info=True)
        trace_event("qa.segment.exception", {"trace_id": trace_id, "tool_name": tool_name, "conversation_id": (tool_args.get("conversation_id") if isinstance(tool_args, dict) else None), "exception": str(ex)})
        return {"segment_committee": {"action": "go", "patch_plan": [], "rationale": f"QA exception: {ex}"}}, []


async def _qa_film2(
    *,
    trace_id: str,
    tool_args: Any,
    last_user_text: str,
    result_obj: Dict[str, Any],
    effective_mode: str,
    absolutize_url: Callable[[str], str],
    profile_name: Optional[str],
    preset: Dict[str, Any],
    state_dir: str,
    tool_runner: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(trace_id, str) or not trace_id:
        log.error("tool_step_qa._qa_film2: missing trace_id - upstream caller must pass trace_id. Continuing with empty trace_id but this is an error.")
        trace_id = ""  # Continue processing but log the error - NO FALLBACK GENERATION
    if not isinstance(result_obj, dict):
        log.warning(f"tool_step_qa._qa_film2: invalid result_obj type={type(result_obj).__name__!r} trace_id={trace_id!r}")
        return []
    try:
        trace_event("qa.film2.start", {"trace_id": trace_id, "conversation_id": (tool_args.get("conversation_id") if isinstance(tool_args, dict) else None)})
        extra_results: List[Dict[str, Any]] = []
        meta_obj = result_obj.get("meta") if isinstance(result_obj.get("meta"), dict) else {}
        refine_budget = _clamp_refine_budget(preset, default=1, cap=1)
        segments_obj = meta_obj.get("segments") if isinstance(meta_obj.get("segments"), dict) else {}
        clips_for_committee = segments_obj.get("clips") if isinstance(segments_obj.get("clips"), list) else None
        if not isinstance(clips_for_committee, list) or not clips_for_committee:
            if clips_for_committee is not None:
                log.debug(f"tool_step_qa._qa_film2: clips_for_committee is not a list or is empty type={type(clips_for_committee).__name__!r} trace_id={trace_id!r}")
            else:
                log.debug(f"tool_step_qa._qa_film2: no clips found in segments_obj trace_id={trace_id!r}")
        if isinstance(clips_for_committee, list) and clips_for_committee:
            seg_results_payload: List[Dict[str, Any]] = []
            conversation_id = tool_args.get("conversation_id") if isinstance(tool_args, dict) else None
            for clip_entry in clips_for_committee:
                if not isinstance(clip_entry, dict):
                    continue
                segment_id = clip_entry.get("segment_id")
                res_obj = clip_entry.get("result")
                if isinstance(segment_id, str) and segment_id and isinstance(res_obj, dict):
                    # Ensure trace_id and conversation_id are propagated to result.meta
                    meta_clip = res_obj.get("meta")
                    if not isinstance(meta_clip, dict):
                        meta_clip = {}
                        res_obj["meta"] = meta_clip
                    if isinstance(meta_clip, dict):
                        if trace_id and not isinstance(meta_clip.get("trace_id"), str):
                            meta_clip["trace_id"] = trace_id
                        if conversation_id and not isinstance(meta_clip.get("conversation_id"), str):
                            meta_clip["conversation_id"] = conversation_id
                    seg_results_payload.append({"segment_id": segment_id, "result": res_obj})
            if seg_results_payload:
                _, seg_outcome, seg_patch = await segment_qa_and_committee(
                    trace_id=trace_id,
                    user_text=tool_step_eval_user_text("film2.run", tool_args, last_user_text),
                    tool_name="film2.run",
                    segment_results=seg_results_payload,
                    mode=effective_mode,
                    absolutize_url=absolutize_url,
                    quality_profile=profile_name,
                    max_refine_passes=refine_budget,
                    tool_runner=tool_runner,
                    state_dir=state_dir,
                )
                meta_obj["segment_committee"] = seg_outcome
                if seg_patch:
                    meta_obj["segment_patch_results"] = seg_patch
                    extra_results.extend(seg_patch)
                    if isinstance(segments_obj, dict):
                        clips_meta = segments_obj.get("clips") if isinstance(segments_obj.get("clips"), list) else None
                        if isinstance(clips_meta, list):
                            for pr in seg_patch:
                                if not isinstance(pr, dict):
                                    continue
                                if (pr.get("tool_name") or pr.get("tool")) != "video.refine.clip":
                                    continue
                                if pr.get("error"):
                                    continue
                                segment_id = pr.get("segment_id")
                                args_used = pr.get("args") if isinstance(pr.get("args"), dict) else {}
                                refine_mode_val = args_used.get("refine_mode")
                                for clip_entry in clips_meta:
                                    clip_segment_id = None
                                    if isinstance(clip_entry, dict) and isinstance(clip_entry.get("segment_id"), str):
                                        clip_segment_id = clip_entry.get("segment_id")
                                    if isinstance(clip_segment_id, str) and clip_segment_id == segment_id:
                                        clip_meta = clip_entry.setdefault("meta", {})
                                        if isinstance(clip_meta, dict):
                                            clip_meta["refined"] = True
                                        if isinstance(refine_mode_val, str) and refine_mode_val:
                                            clip_meta["refine_mode"] = refine_mode_val
        result_obj["meta"] = meta_obj
        trace_event("qa.film2.complete", {"trace_id": trace_id, "conversation_id": (tool_args.get("conversation_id") if isinstance(tool_args, dict) else None), "clips_count": len(clips_for_committee or []), "patch_count": len(extra_results)})
        return extra_results
    except Exception as ex:
        log.error(f"tool_step_qa._qa_film2.exception trace_id={trace_id!r} ex={ex!r}", exc_info=True)
        trace_event("qa.film2.exception", {"trace_id": trace_id, "conversation_id": (tool_args.get("conversation_id") if isinstance(tool_args, dict) else None), "exception": str(ex)})
        return []


async def _qa_single_tool(
    *,
    trace_id: str,
    tool_name: str,
    tool_args: Any,
    last_user_text: str,
    tool_trace_result: Dict[str, Any],
    result_obj: Dict[str, Any],
    effective_mode: str,
    absolutize_url: Callable[[str], str],
    profile_name: Optional[str],
    preset: Dict[str, Any],
    state_dir: str,
    default_budget: int,
    tool_runner: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(result_obj, dict):
        log.warning(f"tool_step_qa._qa_single_tool: invalid result_obj type={type(result_obj).__name__!r} tool_name={tool_name!r} trace_id={trace_id!r}")
        return []
    if not isinstance(tool_trace_result, dict):
        log.warning(f"tool_step_qa._qa_single_tool: invalid tool_trace_result type={type(tool_trace_result).__name__!r} tool_name={tool_name!r} trace_id={trace_id!r}")
        return []
    refine_budget = _clamp_refine_budget(preset, default=default_budget, cap=1)
    meta_patch, extra_results = await _segment_qa_for_single_result(
        trace_id=trace_id,
        tool_name=tool_name,
        tool_args=tool_args,
        last_user_text=last_user_text,
        tool_trace_result=tool_trace_result,
        effective_mode=effective_mode,
        absolutize_url=absolutize_url,
        profile_name=profile_name,
        refine_budget=refine_budget,
        state_dir=state_dir,
        tool_runner=tool_runner,
    )
    meta_obj = result_obj.get("meta") if isinstance(result_obj.get("meta"), dict) else {}
    meta_obj.update(meta_patch)
    result_obj["meta"] = meta_obj
    return extra_results


async def maybe_run_tool_step_segment_qa(
    *,
    trace_id: str,
    tool_name: str,
    tool_args: Any,
    last_user_text: str,
    tool_trace_result: Dict[str, Any],
    result_obj: Dict[str, Any],
    effective_mode: str,
    absolutize_url: Callable[[str], str],
    quality_presets: Dict[str, Dict[str, Any]],
    state_dir: str,
    tool_runner: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Tool-specific glue for segment QA + committee decisions.

    - Mutates `result_obj["meta"]` to attach `segment_committee` and optional patch results.
    - Returns `extra_results` that should be appended to the orchestration `tool_results` list.
    """
    extra_results: List[Dict[str, Any]] = []
    if not isinstance(result_obj, dict):
        # Never silently skip: emit a trace so the pipeline can see that QA could not run.
        if state_dir:
            checkpoints_append_event(
                state_dir,
                trace_id,
                "qa.tool_step.skip",
                {
                    "trace_id": trace_id,
                    "tool_name": tool_name,
                    "tool": tool_name,
                    "reason": "result_obj_not_dict",
                    "result_obj_type": type(result_obj).__name__,
                    "tool_args_type": type(tool_args).__name__,
                },
            )
        return extra_results

    profile_name, preset, _meta_obj = _get_profile_and_preset(result_obj, quality_presets)

    # film2.run: run per-clip segment QA if meta.segments.clips exists
    if tool_name == "film2.run":
        extra_results.extend(
            await _qa_film2(
                trace_id=trace_id,
                tool_args=tool_args,
                last_user_text=last_user_text,
                result_obj=result_obj,
                effective_mode=effective_mode,
                absolutize_url=absolutize_url,
                profile_name=profile_name,
                preset=preset,
                state_dir=state_dir,
                tool_runner=tool_runner,
            )
        )

    # music tools
    elif tool_name in ("music.infinite.windowed", "music.dispatch", "music.variation", "music.mixdown"):
        extra_results.extend(
            await _qa_single_tool(
                trace_id=trace_id,
                tool_name=tool_name,
                tool_args=tool_args,
                last_user_text=last_user_text,
                tool_trace_result=tool_trace_result,
                result_obj=result_obj,
                effective_mode=effective_mode,
                absolutize_url=absolutize_url,
                profile_name=profile_name,
                preset=preset,
                state_dir=state_dir,
                default_budget=1,
                tool_runner=tool_runner,
            )
        )

    # image tools (optional)
    elif isinstance(tool_name, str) and tool_name.startswith("image."):
        extra_results.extend(
            await _qa_single_tool(
                trace_id=trace_id,
                tool_name=tool_name,
                tool_args=tool_args,
                last_user_text=last_user_text,
                tool_trace_result=tool_trace_result,
                result_obj=result_obj,
                effective_mode=effective_mode,
                absolutize_url=absolutize_url,
                profile_name=profile_name,
                preset=preset,
                state_dir=state_dir,
                default_budget=0,
                tool_runner=tool_runner,
            )
        )

    # tts (optional)
    elif tool_name == "tts.speak":
        extra_results.extend(
            await _qa_single_tool(
                trace_id=trace_id,
                tool_name="tts.speak",
                tool_args=tool_args,
                last_user_text=last_user_text,
                tool_trace_result=tool_trace_result,
                result_obj=result_obj,
                effective_mode=effective_mode,
                absolutize_url=absolutize_url,
                profile_name=profile_name,
                preset=preset,
                state_dir=state_dir,
                default_budget=1,
                tool_runner=tool_runner,
            )
        )

    # sfx: metrics sparse, default to 0 refine passes
    elif tool_name == "audio.sfx.compose":
        extra_results.extend(
            await _qa_single_tool(
                trace_id=trace_id,
                tool_name="audio.sfx.compose",
                tool_args=tool_args,
                last_user_text=last_user_text,
                tool_trace_result=tool_trace_result,
                result_obj=result_obj,
                effective_mode=effective_mode,
                absolutize_url=absolutize_url,
                profile_name=profile_name,
                preset=preset,
                state_dir=state_dir,
                default_budget=0,
                tool_runner=tool_runner,
            )
        )

    return extra_results


