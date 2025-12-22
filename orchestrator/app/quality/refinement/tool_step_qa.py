from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ...state.checkpoints import append_event as checkpoints_append_event
from ..refinement.segment_committee import segment_qa_and_committee


def tool_step_eval_user_text(tool_name: str, tool_args: Any, last_user_text: str) -> str:
    """
    Committee/QA prompt seed that is scoped to a SINGLE tool execution step.

    NOTE: This is intentionally lightweight and deterministic; it should not pull
    in orchestrator/main logging utilities to avoid circular imports.
    """
    a = tool_args if isinstance(tool_args, dict) else {"_raw": tool_args}
    goal_hint = ""
    if isinstance(tool_name, str) and tool_name.startswith("image."):
        p = a.get("prompt") or a.get("text") or ""
        if isinstance(p, str) and p.strip():
            goal_hint = f"Generate/refine image(s) to match prompt={p.strip()!r}."
    elif isinstance(tool_name, str) and tool_name.startswith("video."):
        src = a.get("src") or a.get("video") or a.get("url") or ""
        if isinstance(src, str) and src.strip():
            goal_hint = f"Process video src={src.strip()!r}."
    elif isinstance(tool_name, str) and (tool_name.startswith("music.") or tool_name.startswith("tts.") or tool_name.startswith("sfx.")):
        p = a.get("prompt") or a.get("text") or a.get("lyrics") or ""
        if isinstance(p, str) and p.strip():
            goal_hint = f"Generate/refine audio to match prompt/text={p.strip()!r}."
    elif tool_name == "film2.run":
        syn = a.get("synopsis") or a.get("prompt") or ""
        if isinstance(syn, str) and syn.strip():
            goal_hint = f"Generate film output to match synopsis/prompt={syn.strip()!r}."

    # Keep preview safe: only expose shape + common short fields.
    args_preview: Dict[str, Any]
    if not isinstance(a, dict):
        args_preview = {"type": type(a).__name__}
    else:
        keys = sorted([str(k) for k in a.keys()])
        args_preview = {"keys": keys[:64], "has_raw": bool("_raw" in a)}
        for k in (
            "prompt",
            "text",
            "negative",
            "src",
            "url",
            "image_url",
            "video_url",
            "audio_url",
            "segment_id",
            "quality_profile",
            "profile",
        ):
            v = a.get(k)
            if isinstance(v, str):
                args_preview[k] = (v[:240] + "…") if len(v) > 240 else v
            elif isinstance(v, (int, float, bool)) or v is None:
                args_preview[k] = v

    blob = {
        "scope": "tool_step_only",
        "instruction": (
            "Evaluate ONLY whether THIS SINGLE tool call executed correctly and whether its output matches the tool args. "
            "Do NOT evaluate whether the entire user request has been completed. "
            "Do NOT propose unrelated tools; only revise if this tool's own outputs/QA/locks are below thresholds."
        ),
        "tool": tool_name,
        "tool_goal_hint": goal_hint,
        "tool_args_preview": args_preview,
        "original_user_request_preview": (str(last_user_text or "")[:600] + "…")
        if isinstance(last_user_text, str) and len(last_user_text) > 600
        else str(last_user_text or ""),
    }
    return json.dumps(blob, ensure_ascii=False, default=str)


def _clamp_refine_budget(preset: Dict[str, Any], *, default: int, cap: int = 1) -> int:
    raw = preset.get("max_refine_passes", default) if isinstance(preset, dict) else default
    try:
        return min(int(cap), max(0, int(raw)))
    except Exception:
        return min(int(cap), max(0, int(default)))


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
        return extra_results

    meta_obj = result_obj.get("meta") if isinstance(result_obj.get("meta"), dict) else {}
    profile_name = meta_obj.get("quality_profile") if isinstance(meta_obj.get("quality_profile"), str) else None
    preset = quality_presets.get((profile_name or "standard").lower(), quality_presets.get("standard", {}))

    def _emit(kind: str, payload: Dict[str, Any]) -> None:
        if not state_dir:
            return
        if not isinstance(payload, dict):
            return
        checkpoints_append_event(state_dir, trace_id, str(kind or "event"), payload)

    # film2.run: run per-clip segment QA if meta.segments.clips exists
    if tool_name == "film2.run":
        refine_budget = _clamp_refine_budget(preset, default=1, cap=1)
        segments_obj = meta_obj.get("segments") if isinstance(meta_obj.get("segments"), dict) else {}
        clips_for_committee = segments_obj.get("clips") if isinstance(segments_obj.get("clips"), list) else None
        if isinstance(clips_for_committee, list) and clips_for_committee:
            seg_results_payload: List[Dict[str, Any]] = []
            cid_from_args: Optional[str] = None
            if isinstance(tool_args, dict) and isinstance(tool_args.get("cid"), str) and tool_args.get("cid"):
                cid_from_args = str(tool_args.get("cid"))
            for seg in clips_for_committee:
                if not isinstance(seg, dict):
                    continue
                seg_id = seg.get("id")
                res_obj = seg.get("result")
                if isinstance(seg_id, str) and seg_id and isinstance(res_obj, dict):
                    # Ensure per-clip result.meta.cid exists for downstream segment QA / patch enrichment.
                    if cid_from_args:
                        meta_clip = res_obj.get("meta")
                        if not isinstance(meta_clip, dict):
                            meta_clip = {}
                            res_obj["meta"] = meta_clip
                        if isinstance(meta_clip, dict) and not isinstance(meta_clip.get("cid"), str):
                            meta_clip["cid"] = cid_from_args
                    seg_results_payload.append({"name": seg_id, "result": res_obj})
            if seg_results_payload:
                _emit(
                    "film2.segment_qa_start",
                    {"trace_id": trace_id, "tool": "film2.run", "segment_ids": [s.get("name") for s in seg_results_payload if isinstance(s, dict)], "quality_profile": profile_name},
                )
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
                    # Mark refined clips in segments metadata when patch tool is video.refine.clip
                    if isinstance(segments_obj, dict):
                        clips_meta = segments_obj.get("clips") if isinstance(segments_obj.get("clips"), list) else None
                        if isinstance(clips_meta, list):
                            for pr in seg_patch:
                                if not isinstance(pr, dict):
                                    continue
                                if pr.get("tool") != "video.refine.clip":
                                    continue
                                if pr.get("error"):
                                    continue
                                sid = pr.get("segment_id")
                                args_used = pr.get("args") if isinstance(pr.get("args"), dict) else {}
                                refine_mode_val = args_used.get("refine_mode")
                                for clip in clips_meta:
                                    if isinstance(clip, dict) and clip.get("id") == sid:
                                        clip_meta = clip.setdefault("meta", {})
                                        if isinstance(clip_meta, dict):
                                            clip_meta["refined"] = True
                                            if isinstance(refine_mode_val, str) and refine_mode_val:
                                                clip_meta["refine_mode"] = refine_mode_val
                _emit(
                    "film2.segment_qa_result",
                    {"trace_id": trace_id, "tool": "film2.run", "segment_ids": [s.get("name") for s in seg_results_payload if isinstance(s, dict)], "action": seg_outcome.get("action") if isinstance(seg_outcome, dict) else None},
                )
        result_obj["meta"] = meta_obj
        return extra_results

    # music tools
    if tool_name in ("music.infinite.windowed", "music.dispatch", "music.variation", "music.mixdown"):
        refine_budget = _clamp_refine_budget(preset, default=1, cap=1)
        _, seg_outcome, seg_patch = await segment_qa_and_committee(
            trace_id=trace_id,
            user_text=tool_step_eval_user_text(tool_name, tool_args, last_user_text),
            tool_name=tool_name,
            segment_results=[tool_trace_result],
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
        result_obj["meta"] = meta_obj
        return extra_results

    # image tools (optional)
    if isinstance(tool_name, str) and tool_name.startswith("image."):
        refine_budget = _clamp_refine_budget(preset, default=0, cap=1)
        _, seg_outcome, seg_patch = await segment_qa_and_committee(
            trace_id=trace_id,
            user_text=tool_step_eval_user_text(tool_name, tool_args, last_user_text),
            tool_name=tool_name,
            segment_results=[tool_trace_result],
            mode=effective_mode,
            absolutize_url=absolutize_url,
            quality_profile=profile_name,
            max_refine_passes=refine_budget,
            tool_runner=tool_runner if refine_budget > 0 else None,
            state_dir=state_dir,
        )
        meta_obj["segment_committee"] = seg_outcome
        if seg_patch:
            meta_obj["segment_patch_results"] = seg_patch
            extra_results.extend(seg_patch)
        result_obj["meta"] = meta_obj
        return extra_results

    # tts (optional)
    if tool_name == "tts.speak":
        refine_budget = _clamp_refine_budget(preset, default=1, cap=1)
        _, seg_outcome, seg_patch = await segment_qa_and_committee(
            trace_id=trace_id,
            user_text=tool_step_eval_user_text(tool_name, tool_args, last_user_text),
            tool_name=tool_name,
            segment_results=[tool_trace_result],
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
        result_obj["meta"] = meta_obj
        return extra_results

    # sfx: metrics sparse, default to 0 refine passes
    if tool_name == "audio.sfx.compose":
        refine_budget = _clamp_refine_budget(preset, default=0, cap=1)
        _, seg_outcome, seg_patch = await segment_qa_and_committee(
            trace_id=trace_id,
            user_text=tool_step_eval_user_text(tool_name, tool_args, last_user_text),
            tool_name=tool_name,
            segment_results=[tool_trace_result],
            mode=effective_mode,
            absolutize_url=absolutize_url,
            quality_profile=profile_name,
            max_refine_passes=refine_budget,
            tool_runner=tool_runner if refine_budget > 0 else None,
            state_dir=state_dir,
        )
        meta_obj["segment_committee"] = seg_outcome
        if seg_patch:
            meta_obj["segment_patch_results"] = seg_patch
            extra_results.extend(seg_patch)
        result_obj["meta"] = meta_obj
        return extra_results

    return extra_results


