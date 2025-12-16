from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable

import logging

from ..json_parser import JSONParser
from ..datasets.trace import append_sample as _trace_append
from ..locks.runtime import quality_thresholds as _lock_quality_thresholds
from ..pipeline.assets import (
    compute_domain_qa,
    count_images,
    count_video,
    count_audio,
    collect_urls,
)
from ..review.referee import postrun_committee_decide
from ..plan.catalog import PLANNER_VISIBLE_TOOLS

log = logging.getLogger(__name__)


"""
Canonical segment schema and helpers for QA/committee wiring.

SegmentResult is represented as a plain dict with the following expected keys:

{
    "id": str,                  # stable segment id (e.g. "<tool>:<trace_id>:<index>")
    "tool": str,                # tool name, e.g. "image.dispatch"
    "domain": str,              # "image" | "music" | "tts" | "sfx" | "video" | "film2"
    "name": str | None,         # optional human label
    "index": int,               # 0-based index within parent result
    "trace_id": str,            # parent trace identifier for correlation
    "cid": str | None,          # conversation/client id (when available)
    "result": dict,             # raw segment-level tool result slice
    "meta": {
        "locks": dict | None,   # lock bundle snapshot or subset
        "profile": str | None,  # quality profile name
        "timing": {             # optional timing for audio/video
            "start_s": float,
            "end_s": float,
        } | None,
        "extra": dict | None,   # tool-specific metadata
    },
    "qa": {
        "scores": dict,         # per-segment QA metrics
        "summary": str | None,  # optional human-readable QA summary
    },
    "locks": dict | None,       # optional per-segment locks (may mirror meta.locks)
    "artifacts": [
        {
            "kind": str,        # e.g. "image", "audio", "video", "frame"
            "path": str,        # filesystem path or URL
            "extra": dict | None,
        },
        ...
    ],
}
"""


SegmentResult = Dict[str, Any]


def is_valid_segment_result(obj: Any) -> bool:
    """Lightweight validator for SegmentResult dicts."""
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("id"), str):
        return False
    if not isinstance(obj.get("tool"), str):
        return False
    if not isinstance(obj.get("domain"), str):
        return False
    if not isinstance(obj.get("index"), int):
        return False
    result = obj.get("result")
    if not isinstance(result, dict):
        return False
    meta = obj.get("meta")
    if not isinstance(meta, dict):
        return False
    qa = obj.get("qa")
    if not isinstance(qa, dict):
        return False
    if "scores" not in qa or not isinstance(qa.get("scores"), dict):
        return False
    artifacts = obj.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    return True


def assert_valid_segment_result(obj: Any) -> None:
    # Keep this as a no-op validator; callers that depend on a strict check
    # should perform their own assertions where they handle failures.
    if not is_valid_segment_result(obj):
        return None

def _extract_locks_from_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    meta = result.get("meta")
    if isinstance(meta, dict):
        locks = meta.get("locks")
        if isinstance(locks, dict):
            return locks
    return None


def _extract_profile_from_result(result: Dict[str, Any]) -> Optional[str]:
    meta = result.get("meta")
    if isinstance(meta, dict):
        profile = meta.get("quality_profile") or meta.get("profile")
        if isinstance(profile, str) and profile:
            return profile
    return None


def _extract_artifacts(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    arts = result.get("artifacts")
    if isinstance(arts, list):
        for a in arts:
            if not isinstance(a, dict):
                continue
            kind = a.get("kind")
            path = a.get("path") or a.get("view_url") or a.get("url")
            if isinstance(kind, str) and isinstance(path, str) and path:
                artifacts.append(
                    {
                        "kind": kind,
                        "path": path,
                        "extra": a,
                    }
                )
    # Fallback: synthesize a single artifact when tools don't emit an explicit list.
    if not artifacts:
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        candidates = [
            result.get("url"),
            result.get("view_url"),
            result.get("relative_url"),
            result.get("path"),
            meta.get("url"),
            meta.get("view_url"),
            meta.get("relative_url"),
            meta.get("path"),
            result.get("master_uri"),
            result.get("reel_mp4"),
        ]
        best: Optional[str] = None
        for c in candidates:
            if isinstance(c, str) and c.strip():
                best = c.strip()
                break
        if best:
            low = best.lower()
            kind = "file"
            if any(low.endswith(e) for e in (".png", ".jpg", ".jpeg", ".webp", ".gif")):
                kind = "image"
            elif any(low.endswith(e) for e in (".mp4", ".mov", ".mkv", ".webm")):
                kind = "video"
            elif any(low.endswith(e) for e in (".wav", ".mp3", ".flac", ".aac", ".ogg")):
                kind = "audio"
            artifacts.append({"kind": kind, "path": best, "extra": {"synthetic": True}})
    return artifacts


def _base_segment(
    tool_name: str,
    domain: str,
    trace_id: str,
    cid: Optional[str],
    index: int,
    result: Dict[str, Any],
) -> SegmentResult:
    seg_id = f"{tool_name}:{trace_id}:{index}"
    locks = _extract_locks_from_result(result)
    profile = _extract_profile_from_result(result)
    segment: SegmentResult = {
        "id": seg_id,
        "tool": tool_name,
        "domain": domain,
        "name": None,
        "index": index,
        "trace_id": trace_id,
        "cid": cid,
        "result": result,
        "meta": {
            "locks": locks,
            "profile": profile,
            "timing": None,
            "extra": {},
        },
        "qa": {
            "scores": {},
            "summary": None,
        },
        "locks": locks,
        "artifacts": _extract_artifacts(result),
    }
    return segment


def build_image_segments_from_result(
    tool_name: str,
    trace_id: str,
    cid: Optional[str],
    result: Dict[str, Any],
) -> List[SegmentResult]:
    """
    Build SegmentResult list for image tools.

    Image dispatch/refine results follow the canonical shape emitted by `image.dispatch`
    in `routes/toolrun.py`, where:

    - The list of rendered images is exposed as artifact records under
      result["artifacts"] with kind starting with "image".
    - Lock metadata (including any per-entity lock scores) is exposed under
      result["meta"]["locks"].
    - Image QA metrics (including optional per-entity metrics) live under
      result["qa"]["images"] and result["qa"]["images"]["entities"].

    For Step 2 we build one SegmentResult per image artifact, so the committee
    can target specific images for refinement. When no image artifacts are
    present, we fall back to a single segment for the whole result.
    """
    if not isinstance(result, dict):
        return []
    artifacts = result.get("artifacts")
    image_arts: List[Dict[str, Any]] = []
    if isinstance(artifacts, list):
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if isinstance(kind, str) and kind.startswith("image"):
                image_arts.append(item)
    # Fallback: if no explicit image artifacts, treat the whole result as one segment.
    if not image_arts:
        seg = _base_segment(tool_name, "image", trace_id, cid, 0, result)
        return [seg]
    meta_src = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    qa_src = result.get("qa") if isinstance(result.get("qa"), dict) else {}
    ids_src = result.get("ids") if isinstance(result.get("ids"), dict) else {}
    segments: List[SegmentResult] = []
    for idx, art in enumerate(image_arts):
        img_result: Dict[str, Any] = {}
        if isinstance(meta_src, dict):
            meta_copy: Dict[str, Any] = dict(meta_src)
            meta_copy["image_index"] = idx
            art_id = art.get("id")
            if isinstance(art_id, str) and art_id:
                meta_copy["image_id"] = art_id
            img_result["meta"] = meta_copy
        if isinstance(qa_src, dict):
            img_result["qa"] = dict(qa_src)
        if isinstance(ids_src, dict):
            img_result["ids"] = dict(ids_src)
        img_result["artifacts"] = [art]
        seg = _base_segment(tool_name, "image", trace_id, cid, idx, img_result)
        segments.append(seg)
    return segments


def build_music_segments_from_result(
    tool_name: str,
    trace_id: str,
    cid: Optional[str],
    result: Dict[str, Any],
) -> List[SegmentResult]:
    """
    Build SegmentResult list for music tools.

    For mixdown-style tools we emit a single segment for the full track.
    For windowed tools that include a "windows" list in result["meta"]["windows"],
    we emit one segment per window so the committee can reason about and refine
    specific windows.
    """
    if not isinstance(result, dict):
        return []
    meta_obj = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    windows = meta_obj.get("windows") if isinstance(meta_obj.get("windows"), list) else []
    if not windows:
        seg = _base_segment(tool_name, "music", trace_id, cid, 0, result)
        return [seg]
    segments: List[SegmentResult] = []
    for index, win in enumerate(windows):
        if not isinstance(win, dict):
            continue
        window_id = win.get("window_id") or f"win_{index}"
        section_id = win.get("section_id")
        t_start = win.get("t_start")
        t_end = win.get("t_end")
        clip_path = win.get("artifact_path")
        win_meta: Dict[str, Any] = {
            "window_id": window_id,
            "section_id": section_id,
            "timing": {
                "start_s": float(t_start) if isinstance(t_start, (int, float)) else 0.0,
                "end_s": float(t_end) if isinstance(t_end, (int, float)) else 0.0,
            },
            "locks": meta_obj.get("locks") if isinstance(meta_obj.get("locks"), dict) else None,
            "extra": win.get("metrics") if isinstance(win.get("metrics"), dict) else {},
        }
        seg_result: Dict[str, Any] = {
            "meta": win_meta,
            "artifacts": [],
        }
        if isinstance(clip_path, str) and clip_path:
            seg_result["artifacts"] = [
                {
                    "kind": "audio",
                    "path": clip_path,
                    "extra": {"window_id": window_id, "section_id": section_id},
                }
            ]
        seg = _base_segment(tool_name, "music", trace_id, cid, index, seg_result)
        seg['name'] = str(window_id)
        segments.append(seg)
    if not segments:
        seg = _base_segment(tool_name, "music", trace_id, cid, 0, result)
        return [seg]
    return segments


def build_tts_segments_from_result(
    tool_name: str,
    trace_id: str,
    cid: Optional[str],
    result: Dict[str, Any],
) -> List[SegmentResult]:
    """Build SegmentResult list for TTS tools. For now: one segment per full utterance."""
    if not isinstance(result, dict):
        return []
    seg = _base_segment(tool_name, "tts", trace_id, cid, 0, result)
    return [seg]


def build_sfx_segments_from_result(
    tool_name: str,
    trace_id: str,
    cid: Optional[str],
    result: Dict[str, Any],
) -> List[SegmentResult]:
    """Build SegmentResult list for SFX tools. For now: one segment per call."""
    if not isinstance(result, dict):
        return []
    seg = _base_segment(tool_name, "sfx", trace_id, cid, 0, result)
    return [seg]


def build_video_segments_from_result(
    tool_name: str,
    trace_id: str,
    cid: Optional[str],
    result: Dict[str, Any],
) -> List[SegmentResult]:
    """Build SegmentResult list for video/film2 tools. For now: one segment per call."""
    if not isinstance(result, dict):
        return []
    domain = "film2" if tool_name == "film2.run" else "video"
    seg = _base_segment(tool_name, domain, trace_id, cid, 0, result)
    return [seg]


def build_segments_for_tool(
    tool_name: str,
    *,
    trace_id: str,
    cid: Optional[str],
    result: Dict[str, Any],
) -> List[SegmentResult]:
    """Dispatch to the appropriate builder based on tool name."""
    if not isinstance(tool_name, str):
        return []
    if tool_name.startswith("image."):
        return build_image_segments_from_result(tool_name, trace_id, cid, result)
    if tool_name.startswith("music."):
        return build_music_segments_from_result(tool_name, trace_id, cid, result)
    if tool_name == "tts.speak":
        return build_tts_segments_from_result(tool_name, trace_id, cid, result)
    if tool_name == "audio.sfx.compose":
        return build_sfx_segments_from_result(tool_name, trace_id, cid, result)
    if tool_name.startswith("video.") or tool_name == "film2.run":
        return build_video_segments_from_result(tool_name, trace_id, cid, result)
    # Fallback: treat as generic segment with unknown domain
    return build_video_segments_from_result(tool_name, trace_id, cid, result)


# Patch tool whitelist for committee patch plans.
ALLOWED_PATCH_TOOLS = {
    "image.refine.segment",
    "video.refine.clip",
    "music.refine.window",
    "tts.refine.segment",
}


def filter_patch_plan(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """Filter raw patch plan down to allowed tools and known segment ids."""
    segment_ids = {seg.get("id") for seg in segments if isinstance(seg, dict)}
    filtered: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            continue
        step_tool = (step.get("tool") or "").strip()
        if not step_tool or step_tool not in ALLOWED_PATCH_TOOLS:
            continue
        seg_id = step.get("segment_id")
        # segment_id is mandatory for all allowed patch tools and must match an existing segment
        if not isinstance(seg_id, str) or not seg_id:
            continue
        if seg_id not in segment_ids:
            continue
        args_val = step.get("args")
        if args_val is None and ("arguments" in step):
            args_val = step.get("arguments")
        # IMPORTANT: Do not drop args if they arrive as a JSON string.
        # Normalize into a dict; if parsing fails, preserve under "_raw".
        args: Dict[str, Any]
        if isinstance(args_val, dict):
            args = dict(args_val)
        elif isinstance(args_val, str):
            parsed = JSONParser().parse(args_val or "", dict)
            args = dict(parsed) if isinstance(parsed, dict) else {"_raw": args_val}
        elif args_val is None:
            args = {}
        else:
            args = {"_raw": args_val}
        filtered.append({"tool": step_tool, "segment_id": seg_id, "args": args})
    return filtered


def enrich_patch_plan_for_tts_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich tts.refine.segment steps with src/source audio + lock_bundle derived from TTS segments.
    """
    index_by_id: Dict[str, SegmentResult] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_id = seg.get("id")
        domain = seg.get("domain")
        if isinstance(seg_id, str) and seg_id and isinstance(domain, str) and domain == "tts":
            index_by_id[seg_id] = seg
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            enriched.append(step)
            continue
        tool_name = (step.get("tool") or "").strip()
        seg_id = step.get("segment_id")
        args_obj = step.get("args") if isinstance(step.get("args"), dict) else {}
        if tool_name != "tts.refine.segment":
            enriched.append(step)
            continue
        if not isinstance(seg_id, str) or not seg_id:
            enriched.append(step)
            continue
        seg = index_by_id.get(seg_id)
        if not isinstance(seg, dict):
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        if "segment_id" not in args:
            args["segment_id"] = seg_id
        seg_result = seg.get("result") if isinstance(seg.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        artifacts = seg_result.get("artifacts") if isinstance(seg_result.get("artifacts"), list) else []
        if "src" not in args and "source_audio" not in args:
            src_val: Optional[str] = None
            if artifacts:
                first_art = artifacts[0]
                if isinstance(first_art, dict):
                    path_val = first_art.get("path")
                    view_val = first_art.get("view_url")
                    if isinstance(path_val, str) and path_val.strip():
                        src_val = path_val
                    elif isinstance(view_val, str) and view_val.strip():
                        src_val = view_val
            if isinstance(src_val, str) and src_val:
                args["src"] = src_val
        if "lock_bundle" not in args:
            locks_val = meta.get("locks")
            if isinstance(locks_val, dict):
                inner = locks_val.get("bundle")
                args["lock_bundle"] = inner if isinstance(inner, dict) else locks_val
            elif isinstance(seg.get("locks"), dict):
                args["lock_bundle"] = seg.get("locks")
        if "cid" not in args:
            cid_val = seg.get("cid")
            if isinstance(cid_val, str) and cid_val:
                args["cid"] = cid_val
        enriched.append({"tool": tool_name, "segment_id": seg_id, "args": args})
    return enriched


def enrich_patch_plan_for_sfx_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich audio.sfx.compose patch steps (if ever used) with src + lock_bundle.
    """
    index_by_id: Dict[str, SegmentResult] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_id = seg.get("id")
        domain = seg.get("domain")
        if isinstance(seg_id, str) and seg_id and isinstance(domain, str) and domain == "sfx":
            index_by_id[seg_id] = seg
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            enriched.append(step)
            continue
        tool_name = (step.get("tool") or "").strip()
        seg_id = step.get("segment_id")
        args_obj = step.get("args") if isinstance(step.get("args"), dict) else {}
        if tool_name != "audio.sfx.compose":
            enriched.append(step)
            continue
        if not isinstance(seg_id, str) or not seg_id:
            enriched.append(step)
            continue
        seg = index_by_id.get(seg_id)
        if not isinstance(seg, dict):
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        if "segment_id" not in args:
            args["segment_id"] = seg_id
        seg_result = seg.get("result") if isinstance(seg.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        if "lock_bundle" not in args and isinstance(meta.get("locks"), dict):
            locks_val = meta.get("locks")
            inner = locks_val.get("bundle")
            args["lock_bundle"] = inner if isinstance(inner, dict) else locks_val
        if "cid" not in args:
            cid_val = seg.get("cid")
            if isinstance(cid_val, str) and cid_val:
                args["cid"] = cid_val
        enriched.append({"tool": tool_name, "segment_id": seg_id, "args": args})
    return enriched


def enrich_patch_plan_for_image_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich image.refine.segment steps with segment_id, prompt, source_image,
    and lock_bundle derived from the corresponding image segments.
    """
    index_by_id: Dict[str, SegmentResult] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_id = seg.get("id")
        domain = seg.get("domain")
        if isinstance(seg_id, str) and seg_id and isinstance(domain, str) and domain == "image":
            index_by_id[seg_id] = seg
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            continue
        tool_name = (step.get("tool") or "").strip()
        seg_id = step.get("segment_id")
        args_obj = step.get("args") if isinstance(step.get("args"), dict) else {}
        if tool_name != "image.refine.segment":
            enriched.append(step)
            continue
        if not isinstance(seg_id, str) or not seg_id:
            enriched.append(step)
            continue
        seg = index_by_id.get(seg_id)
        if not isinstance(seg, dict):
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        # Ensure the refine tool receives the segment identifier in-arguments.
        # The executor for image.refine.segment validates segment_id from args
        # rather than the outer patch-plan field, so we mirror it here.
        if "segment_id" not in args:
            args["segment_id"] = seg_id
        seg_result = seg.get("result") if isinstance(seg.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        artifacts = seg_result.get("artifacts") if isinstance(seg_result.get("artifacts"), list) else []
        # prompt from meta
        if "prompt" not in args:
            prompt_val = meta.get("prompt")
            if isinstance(prompt_val, str) and prompt_val.strip():
                args["prompt"] = prompt_val
        # source_image from first artifact path or view_url
        if "source_image" not in args:
            source_image_val: Optional[str] = None
            if artifacts:
                first_art = artifacts[0]
                if isinstance(first_art, dict):
                    path_val = first_art.get("path")
                    view_val = first_art.get("view_url")
                    if isinstance(path_val, str) and path_val.strip():
                        source_image_val = path_val
                    elif isinstance(view_val, str) and view_val.strip():
                        source_image_val = view_val
            if isinstance(source_image_val, str) and source_image_val:
                args["source_image"] = source_image_val
        # lock_bundle from meta.locks
        if "lock_bundle" not in args:
            locks = meta.get("locks")
            bundle_candidate: Optional[Dict[str, Any]] = None
            if isinstance(locks, dict):
                inner_bundle = locks.get("bundle")
                if isinstance(inner_bundle, dict):
                    bundle_candidate = inner_bundle
                else:
                    bundle_candidate = locks
            if isinstance(bundle_candidate, dict):
                args["lock_bundle"] = bundle_candidate
        enriched.append({"tool": tool_name, "segment_id": seg_id, "args": args})
    return enriched


def enrich_patch_plan_for_video_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich video.refine.clip steps with src, timecode, prompt, and locks
    derived from the corresponding video clip segments.
    """
    index_by_id: Dict[str, SegmentResult] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_id = seg.get("id")
        domain = seg.get("domain")
        if isinstance(seg_id, str) and seg_id and isinstance(domain, str) and domain in ("video", "film2"):
            index_by_id[seg_id] = seg
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            continue
        tool_name = (step.get("tool") or "").strip()
        seg_id = step.get("segment_id")
        args_obj = step.get("args") if isinstance(step.get("args"), dict) else {}
        if tool_name != "video.refine.clip":
            enriched.append(step)
            continue
        if not isinstance(seg_id, str) or not seg_id:
            enriched.append(step)
            continue
        seg = index_by_id.get(seg_id)
        if not isinstance(seg, dict):
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        seg_result = seg.get("result") if isinstance(seg.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        artifacts = seg_result.get("artifacts") if isinstance(seg_result.get("artifacts"), list) else []
        # src from first artifact path or view_url
        if "src" not in args:
            src_val: Optional[str] = None
            if artifacts:
                first_art = artifacts[0]
                if isinstance(first_art, dict):
                    path_val = first_art.get("path")
                    view_val = first_art.get("view_url")
                    if isinstance(path_val, str) and path_val.strip():
                        src_val = path_val
                    elif isinstance(view_val, str) and view_val.strip():
                        src_val = view_val
            if isinstance(src_val, str) and src_val:
                args["src"] = src_val
        # timecode from meta.timecode
        if "timecode" not in args:
            timecode = meta.get("timecode")
            if isinstance(timecode, dict):
                args["timecode"] = timecode
        # prompt from meta.prompt if present
        if "prompt" not in args:
            prompt_val = meta.get("prompt")
            if isinstance(prompt_val, str) and prompt_val.strip():
                args["prompt"] = prompt_val
        # locks from meta.locks
        if "locks" not in args:
            locks_val = meta.get("locks")
            if isinstance(locks_val, dict):
                args["locks"] = locks_val
        # propagate width/height if present
        if "width" not in args:
            width_val = meta.get("width")
            if isinstance(width_val, (int, float)):
                args["width"] = int(width_val)
        if "height" not in args:
            height_val = meta.get("height")
            if isinstance(height_val, (int, float)):
                args["height"] = int(height_val)
        # propagate identifiers for richer tracing
        if "film_id" not in args:
            film_id = meta.get("film_id")
            if isinstance(film_id, str) and film_id.strip():
                args["film_id"] = film_id
        if "scene_id" not in args:
            scene_id = meta.get("scene_id")
            if isinstance(scene_id, str) and scene_id.strip():
                args["scene_id"] = scene_id
        if "shot_id" not in args:
            shot_id = meta.get("shot_id")
            if isinstance(shot_id, str) and shot_id.strip():
                args["shot_id"] = shot_id
        enriched.append({"tool": tool_name, "segment_id": seg_id, "args": args})
    return enriched


def enrich_patch_plan_for_music_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich music.refine.window steps with window/audio context derived from
    the corresponding music segments.
    """
    index_by_id: Dict[str, SegmentResult] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_id = seg.get("id")
        domain = seg.get("domain")
        if isinstance(seg_id, str) and seg_id and isinstance(domain, str) and domain == "music":
            index_by_id[seg_id] = seg
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            enriched.append(step)
            continue
        tool_name = (step.get("tool") or "").strip()
        seg_id = step.get("segment_id")
        args_obj = step.get("args") if isinstance(step.get("args"), dict) else {}
        if tool_name != "music.refine.window":
            enriched.append(step)
            continue
        if not isinstance(seg_id, str) or not seg_id:
            enriched.append(step)
            continue
        seg = index_by_id.get(seg_id)
        if not isinstance(seg, dict):
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        seg_result = seg.get("result") if isinstance(seg.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        artifacts = seg_result.get("artifacts") if isinstance(seg_result.get("artifacts"), list) else []
        if "segment_id" not in args:
            args["segment_id"] = seg_id
        window_id_val = meta.get("window_id")
        if isinstance(window_id_val, str) and window_id_val and "window_id" not in args:
            args["window_id"] = window_id_val
        if "src" not in args:
            src_val = None
            if artifacts:
                first_art = artifacts[0]
                if isinstance(first_art, dict):
                    path_val = first_art.get("path")
                    view_val = first_art.get("view_url")
                    if isinstance(path_val, str) and path_val.strip():
                        src_val = path_val
                    elif isinstance(view_val, str) and view_val.strip():
                        src_val = view_val
            if isinstance(src_val, str) and src_val:
                args["src"] = src_val
        # Propagate lock bundle and cid where available so the refine tool can
        # update mappings and optionally restitch a full track.
        if "lock_bundle" not in args:
            locks_val = seg.get("locks")
            if isinstance(locks_val, dict):
                args["lock_bundle"] = locks_val
        if "cid" not in args:
            cid_val = seg.get("cid")
            if isinstance(cid_val, str) and cid_val:
                args["cid"] = cid_val
        enriched.append({"tool": tool_name, "segment_id": seg_id, "args": args})
    return enriched


async def apply_patch_plan(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
    tool_runner,
    trace_id: Optional[str],
    parent_tool: Optional[str],
) -> (List[SegmentResult], List[Dict[str, Any]]):
    """
    Apply a filtered patch plan to segments using the provided tool_runner.

    tool_runner should be an async callable taking a {"name":..., "arguments":...}
    dict and returning a result dict (the existing execute_tool_call is expected).
    """
    index_by_id: Dict[str, int] = {}
    for idx, seg in enumerate(segments):
        if isinstance(seg, dict) and isinstance(seg.get("id"), str):
            index_by_id[seg["id"]] = idx
    updated_segments: List[SegmentResult] = list(segments)
    # IMPORTANT:
    # `patch_results` must be shaped like normal tool results so downstream code
    # (QA aggregation, asset URL collection, finalize) can treat them uniformly:
    #   {"name": str, "result": dict, "args": dict, "error": dict|str|None, ...}
    patch_results: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        tool_name = step.get("tool")
        seg_id = step.get("segment_id")
        args = step.get("args")
        if not isinstance(tool_name, str) or not tool_name:
            continue
        if not isinstance(args, dict):
            continue
        seg_index = index_by_id.get(seg_id) if isinstance(seg_id, str) else None
        target_segment: Optional[SegmentResult] = None
        if isinstance(seg_index, int) and 0 <= seg_index < len(updated_segments):
            target_segment = updated_segments[seg_index]
        tool_call = {"name": tool_name, "arguments": args}
        # tool_runner returns a tool-result shaped dict (typically via executor_gateway):
        # {"name":..., "result": {...}, "args": {...}, "error": {...}|None, "step_id": ...}
        tool_env = await tool_runner(tool_call)
        error_obj: Any = None
        res_obj: Dict[str, Any] = {}
        step_id_val: Optional[str] = None
        args_echo: Dict[str, Any] = dict(args)
        if isinstance(tool_env, dict):
            error_obj = tool_env.get("error")
            step_id_val = tool_env.get("step_id") if isinstance(tool_env.get("step_id"), str) else None
            args_in = tool_env.get("args")
            if isinstance(args_in, dict) and args_in:
                args_echo = dict(args_in)
            inner = tool_env.get("result")
            if isinstance(inner, dict):
                res_obj = inner
        patch_entry: Dict[str, Any] = {
            # Tool-result canonical shape
            "name": str(tool_name),
            "result": res_obj,
            "args": args_echo,
            "error": error_obj,
            "step_id": step_id_val,
            # Patch metadata (extra fields are tolerated throughout the pipeline)
            "segment_id": seg_id,
            "tool": tool_name,
            "parent_tool": parent_tool,
        }
        patch_results.append(patch_entry)
        if (
            target_segment is not None
            and isinstance(tool_env, dict)
            and not bool(error_obj)
        ):
            if isinstance(res_obj, dict):
                target_segment["result"] = res_obj
                # Re-hydrate derived segment fields so later enrich/QA phases see the new artifacts/locks.
                try:
                    target_segment["artifacts"] = _extract_artifacts(res_obj)
                    new_locks = _extract_locks_from_result(res_obj)
                    if isinstance(new_locks, dict):
                        target_segment["locks"] = new_locks
                        meta_obj = target_segment.get("meta") if isinstance(target_segment.get("meta"), dict) else {}
                        if isinstance(meta_obj, dict):
                            meta_obj["locks"] = new_locks
                            target_segment["meta"] = meta_obj
                    new_profile = _extract_profile_from_result(res_obj)
                    if isinstance(new_profile, str) and new_profile:
                        meta_obj = target_segment.get("meta") if isinstance(target_segment.get("meta"), dict) else {}
                        if isinstance(meta_obj, dict):
                            meta_obj["profile"] = new_profile
                            target_segment["meta"] = meta_obj
                except Exception:
                    # Never allow hydration to break patch execution
                    log.debug("qa.segments: failed to hydrate derived segment fields (non-fatal)", exc_info=True)
                # Emit a refine trace row for successful segment updates
                try:
                    if isinstance(trace_id, str) and trace_id.strip():
                        seg_domain = target_segment.get("domain")
                        seg_meta = target_segment.get("meta") if isinstance(target_segment.get("meta"), dict) else {}
                        seg_locks = target_segment.get("locks") if isinstance(target_segment.get("locks"), dict) else None
                        cid_val = target_segment.get("cid")
                        new_artifacts = res_obj.get("artifacts") if isinstance(res_obj.get("artifacts"), list) else []
                        artifact_path: Optional[str] = None
                        artifact_view: Optional[str] = None
                        if new_artifacts:
                            first_art = new_artifacts[0]
                            if isinstance(first_art, dict):
                                path_val = first_art.get("path")
                                view_val = first_art.get("view_url")
                                if isinstance(path_val, str) and path_val.strip():
                                    artifact_path = path_val
                                if isinstance(view_val, str) and view_val.strip():
                                    artifact_view = view_val
                        row: Dict[str, Any] = {
                            "trace_id": trace_id,
                            "event": "refine",
                            "parent_tool": parent_tool,
                            "patch_tool": tool_name,
                            "segment_id": seg_id,
                            "domain": seg_domain,
                            "cid": cid_val,
                            "args": args_echo,
                            "locks": seg_locks,
                            "result_meta": res_obj.get("meta") if isinstance(res_obj.get("meta"), dict) else {},
                        }
                        if artifact_path is not None:
                            row["artifact_path"] = artifact_path
                        if artifact_view is not None:
                            row["artifact_view_url"] = artifact_view
                        _trace_append("segments", row)
                except Exception:
                    # Tracing must not break patch execution
                    log.debug("qa.segments: failed to append refine trace row (non-fatal)", exc_info=True)
    return updated_segments, patch_results


