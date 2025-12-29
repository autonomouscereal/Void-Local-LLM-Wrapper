from __future__ import annotations

"""
quality.segments: Canonical segment QA helpers for committee/refinement wiring.

This module replaces the legacy path `app.qa.segments` (no shims).
"""

from typing import Any, Dict, List, Optional, Tuple, Callable

import copy
import logging
import os

from ..json_parser import JSONParser
from ..tracing.runtime import trace_event
from ..locks.runtime import quality_thresholds as _lock_quality_thresholds
from .metrics.tool_results import (
    compute_domain_qa,
    count_images,
    count_video,
    count_audio,
    collect_urls,
)
from ..quality.decisions.committee import postrun_committee_decide
from ..plan.catalog import PLANNER_VISIBLE_TOOLS
from void_artifacts import build_artifact, Artifact, generate_artifact_id
import time

log = logging.getLogger(__name__)


"""
Canonical segment schema and helpers for QA/committee wiring.

SegmentResult is represented as a plain dict with the following expected keys:

{
    "segment_id": str,          # stable segment id (e.g. "<tool>:<trace_id>:<index>")
    "tool_name": str,          # tool name, e.g. "image.dispatch"
    "tool": str,                # tool name (backward compat), e.g. "image.dispatch"
    "domain": str,              # "image" | "music" | "tts" | "sfx" | "video" | "film2"
    "name": str | None,         # optional human label
    "index": int,               # 0-based index within parent result
    "trace_id": str,            # parent trace identifier for correlation
    "conversation_id": str | None,          # conversation/client id (when available)
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
    ok = True
    reasons: List[str] = []
    if not isinstance(obj, dict):
        ok = False
        reasons.append(f"type={type(obj).__name__}")
    else:
        if not isinstance(obj.get("segment_id"), str):
            ok = False
            reasons.append("missing_segment_id")
        if not isinstance(obj.get("tool"), str):
            ok = False
            reasons.append("missing_tool")
        if not isinstance(obj.get("domain"), str):
            ok = False
            reasons.append("missing_domain")
        if not isinstance(obj.get("index"), int):
            ok = False
            reasons.append("missing_index")
        result = obj.get("result")
        if not isinstance(result, dict):
            ok = False
            reasons.append("missing_result_dict")
        meta = obj.get("meta")
        if not isinstance(meta, dict):
            ok = False
            reasons.append("missing_meta_dict")
        qa = obj.get("qa")
        if not isinstance(qa, dict):
            ok = False
            reasons.append("missing_qa_dict")
        else:
            if "scores" not in qa or not isinstance(qa.get("scores"), dict):
                ok = False
                reasons.append("missing_qa_scores")
        artifacts = obj.get("artifacts")
        if not isinstance(artifacts, list):
            ok = False
            reasons.append("missing_artifacts_list")
    if not ok:
        log.warning(f"segments.is_valid_segment_result invalid reasons={reasons} obj_type={type(obj).__name__}")
    return bool(ok)


def assert_valid_segment_result(obj: Any) -> None:
    # Keep this as a no-op validator; callers that depend on a strict check
    # should perform their own assertions where they handle failures.
    if not is_valid_segment_result(obj):
        log.warning(f"segments.assert_valid_segment_result invalid_segment obj_type={type(obj).__name__!r}")
    # Intentionally no return value.


def _extract_locks_from_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    locks: Optional[Dict[str, Any]] = None
    meta = result.get("meta")
    if not isinstance(meta, dict):
        log.debug(f"segments._extract_locks_from_result missing_meta result_keys={sorted([str(k) for k in (result or {}).keys()]) if isinstance(result, dict) else type(result).__name__}")
    else:
        l = meta.get("locks")
        if isinstance(l, dict):
            locks = l
        else:
            log.debug(f"segments._extract_locks_from_result no_locks meta_keys={sorted([str(k) for k in meta.keys()])[:128]}")
    return locks


def _extract_profile_from_result(result: Dict[str, Any]) -> Optional[str]:
    profile_out: Optional[str] = None
    meta = result.get("meta")
    if not isinstance(meta, dict):
        log.debug(f"segments._extract_profile_from_result missing_meta result_keys={sorted([str(k) for k in (result or {}).keys()]) if isinstance(result, dict) else type(result).__name__}")
    else:
        profile = meta.get("quality_profile") or meta.get("profile")
        if isinstance(profile, str) and profile:
            profile_out = profile
    return profile_out


def _extract_artifacts(result: Dict[str, Any], *, trace_id: Optional[str] = None, conversation_id: Optional[str] = None, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract artifacts from result, ensuring all required fields are present.
    
    Propagates trace_id, conversation_id, and tool_name from result.meta if not provided.
    """
    artifacts: List[Dict[str, Any]] = []
    # Extract from result.meta if not provided
    meta_obj = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    trace_id = trace_id or meta_obj.get("trace_id") or ""
    conversation_id = conversation_id or meta_obj.get("conversation_id")
    tool_name = tool_name or meta_obj.get("tool_name") or meta_obj.get("tool")
    
    artifacts_list = result.get("artifacts")
    if isinstance(artifacts_list, list):
        for a in artifacts_list:
            if not isinstance(a, dict):
                log.debug("_extract_artifacts: skipping non-dict artifact trace_id=%s tool_name=%s type=%s", trace_id, tool_name, type(a).__name__)
                continue
            kind = a.get("kind")
            path = a.get("path") or a.get("view_url") or a.get("url")
            if isinstance(kind, str) and isinstance(path, str) and path:
                artifact_id = a.get("artifact_id")
                if not artifact_id and path:
                    # Generate artifact_id if missing (for backward compatibility)
                    artifact_id = generate_artifact_id(
                        trace_id=(trace_id or ""),
                        tool_name=(tool_name or "unknown"),
                        conversation_id=(conversation_id or ""),
                        suffix_data=os.path.basename(path),
                    )
                # Ensure artifact has required meta fields
                a_meta = a.get("meta") if isinstance(a.get("meta"), dict) else {}
                if trace_id and not isinstance(a_meta.get("trace_id"), str):
                    a_meta["trace_id"] = trace_id
                if conversation_id and not isinstance(a_meta.get("conversation_id"), str):
                    a_meta["conversation_id"] = conversation_id
                if tool_name and not isinstance(a_meta.get("tool_name"), str):
                    a_meta["tool_name"] = tool_name
                    a_meta["tool"] = tool_name
                if not isinstance(a_meta.get("created_ms"), int):
                    import time
                    a_meta["created_ms"] = int(time.time() * 1000)
                    a_meta["created_at"] = int(time.time())
                # Use Artifact.from_dict to ensure proper structure
                artifact_dict = {"artifact_id": artifact_id, "kind": kind, "path": path, "meta": a_meta, "extra": a}
                artifact = Artifact.from_dict(artifact_dict, trace_id=trace_id, conversation_id=conversation_id, tool_name=tool_name)
                artifacts.append(artifact.to_dict())
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
            # Generate unique artifact_id for synthetic artifact
            artifact_id = None
            if best:
                artifact_id = generate_artifact_id(
                    trace_id=(trace_id or ""),
                    tool_name=(tool_name or "unknown"),
                    conversation_id=(conversation_id or ""),
                    suffix_data=os.path.basename(best),
                )
            # Use build_artifact to ensure proper structure for synthetic artifacts
            artifacts.append(
                build_artifact(
                    artifact_id=artifact_id or "unknown",
                    kind=kind,
                    path=best,
                    trace_id=trace_id or "",
                    conversation_id=conversation_id,
                    tool_name=tool_name,
                    extra={"synthetic": True},
                )
            )
    return artifacts


def _base_segment(*, tool_name: str, domain: str, trace_id: str, conversation_id: str, index: int, result: Dict[str, Any]) -> SegmentResult:
    segment_id = f"{tool_name}:{trace_id}:{index}"
    locks = _extract_locks_from_result(result)
    profile = _extract_profile_from_result(result)
    segment: SegmentResult = {
        "segment_id": segment_id,
        "tool_name": tool_name,
        "tool": tool_name,
        "domain": domain,
        "name": None,
        "index": index,
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "result": result,
        "meta": {"locks": locks, "profile": profile, "timing": None, "extra": {}},
        "qa": {"scores": {}, "summary": None},
        "locks": locks,
        "artifacts": _extract_artifacts(result, trace_id=trace_id, conversation_id=conversation_id, tool_name=tool_name),
    }
    return segment


def build_image_segments_from_result(*, tool_name: str, trace_id: str, conversation_id: str, result: Dict[str, Any]) -> List[SegmentResult]:
    segments: List[SegmentResult] = []
    if not isinstance(result, dict):
        log.warning(f"segments.build_image_segments_from_result invalid_result tool={tool_name!r} result_type={type(result).__name__!r}")
        err_result: Dict[str, Any] = {
            "meta": {"conversation_id": conversation_id, "trace_id": trace_id, "error": {"code": "invalid_result", "message": "result must be dict", "type": type(result).__name__}},
            "artifacts": [],
        }
        segments = segments + [_base_segment(tool_name=tool_name, domain="image", trace_id=trace_id, conversation_id=conversation_id, index=0, result=err_result)]
        return segments
    artifacts = result.get("artifacts")
    image_arts: List[Dict[str, Any]] = []
    if isinstance(artifacts, list):
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if isinstance(kind, str) and kind.startswith("image"):
                image_arts.append(item)
    if not image_arts:
        log.warning(f"segments.build_image_segments_from_result no_image_artifacts tool={tool_name!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
        segments = segments + [_base_segment(tool_name=tool_name, domain="image", trace_id=trace_id, conversation_id=conversation_id, index=0, result=result)]
        return segments
    meta_src = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    qa_src = result.get("qa") if isinstance(result.get("qa"), dict) else {}
    external_ids = result.get("ids") if isinstance(result.get("ids"), dict) else {}  # external API identifiers
    for idx, artifact in enumerate(image_arts):
        img_result: Dict[str, Any] = {}
        if isinstance(meta_src, dict):
            meta_copy: Dict[str, Any] = dict(meta_src)
            meta_copy["image_index"] = idx
            artifact_id = artifact.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id:
                meta_copy["image_id"] = artifact_id
            img_result["meta"] = meta_copy
        if isinstance(qa_src, dict):
            img_result["qa"] = dict(qa_src)
        if isinstance(external_ids, dict):
            img_result["ids"] = dict(external_ids)  # external API identifiers
        img_result["artifacts"] = [artifact]
        segments = segments + [_base_segment(tool_name=tool_name, domain="image", trace_id=trace_id, conversation_id=conversation_id, index=idx, result=img_result)]
    return segments


def build_music_segments_from_result(*, tool_name: str, trace_id: str, conversation_id: str, result: Dict[str, Any]) -> List[SegmentResult]:
    segments: List[SegmentResult] = []
    if not isinstance(result, dict):
        log.warning(f"segments.build_music_segments_from_result invalid_result tool={tool_name!r} result_type={type(result).__name__!r}")
        err_result: Dict[str, Any] = {
            "meta": {"conversation_id": conversation_id, "trace_id": trace_id, "error": {"code": "invalid_result", "message": "result must be dict", "type": type(result).__name__}},
            "artifacts": [],
        }
        segments = segments + [_base_segment(tool_name=tool_name, domain="music", trace_id=trace_id, conversation_id=conversation_id, index=0, result=err_result)]
        return segments
    meta_obj = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    windows = meta_obj.get("windows") if isinstance(meta_obj.get("windows"), list) else []
    if not windows:
        log.warning(f"segments.build_music_segments_from_result no_windows tool={tool_name!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
        segments = segments + [_base_segment(tool_name=tool_name, domain="music", trace_id=trace_id, conversation_id=conversation_id, index=0, result=result)]
        return segments
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
        seg_result: Dict[str, Any] = {"meta": win_meta, "artifacts": []}
        if isinstance(clip_path, str) and clip_path:
            artifact_id = generate_artifact_id(
                trace_id=(trace_id or ""),
                tool_name=(tool_name or "music.window"),
                conversation_id=(conversation_id or ""),
                suffix_data=f"{window_id}:{os.path.basename(clip_path)}",
            )
            seg_result["artifacts"] = [
                build_artifact(
                    artifact_id=artifact_id,
                    kind="audio",
                    path=clip_path,
                    trace_id=(trace_id or ""),
                    conversation_id=(conversation_id or ""),
                    tool_name=(tool_name or "music.window"),
                    tags=[],
                    extra={"window_id": window_id, "section_id": section_id},
                ).to_dict()
            ]
        segment = _base_segment(tool_name=tool_name, domain="music", trace_id=trace_id, conversation_id=conversation_id, index=index, result=seg_result)
        segment["name"] = str(window_id)
        segments = segments + [segment]
    if not segments:
        log.warning(f"segments.build_music_segments_from_result windows_present_but_no_segments tool={tool_name!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
        segments = segments + [_base_segment(tool_name=tool_name, domain="music", trace_id=trace_id, conversation_id=conversation_id, index=0, result=result)]
    return segments


def build_tts_segments_from_result(*, tool_name: str, trace_id: str, conversation_id: str, result: Dict[str, Any]) -> List[SegmentResult]:
    segments: List[SegmentResult] = []
    if not isinstance(result, dict):
        log.warning(f"segments.build_tts_segments_from_result invalid_result tool={tool_name!r} result_type={type(result).__name__!r}")
        err_result: Dict[str, Any] = {
            "meta": {"conversation_id": conversation_id, "trace_id": trace_id, "error": {"code": "invalid_result", "message": "result must be dict", "type": type(result).__name__}},
            "artifacts": [],
        }
        segments = segments + [_base_segment(tool_name=tool_name, domain="tts", trace_id=trace_id, conversation_id=conversation_id, index=0, result=err_result)]
    else:
        segments = segments + [_base_segment(tool_name=tool_name, domain="tts", trace_id=trace_id, conversation_id=conversation_id, index=0, result=result)]
    return segments


def build_sfx_segments_from_result(*, tool_name: str, trace_id: str, conversation_id: str, result: Dict[str, Any]) -> List[SegmentResult]:
    segments: List[SegmentResult] = []
    if not isinstance(result, dict):
        log.warning(f"segments.build_sfx_segments_from_result invalid_result tool={tool_name!r} result_type={type(result).__name__!r}")
        err_result: Dict[str, Any] = {
            "meta": {"conversation_id": conversation_id, "trace_id": trace_id, "error": {"code": "invalid_result", "message": "result must be dict", "type": type(result).__name__}},
            "artifacts": [],
        }
        segments = segments + [_base_segment(tool_name=tool_name, domain="sfx", trace_id=trace_id, conversation_id=conversation_id, index=0, result=err_result)]
    else:
        segments = segments + [_base_segment(tool_name=tool_name, domain="sfx", trace_id=trace_id, conversation_id=conversation_id, index=0, result=result)]
    return segments


def build_video_segments_from_result(*, tool_name: str, trace_id: str, conversation_id: str, result: Dict[str, Any]) -> List[SegmentResult]:
    segments_out: List[SegmentResult] = []
    if not isinstance(result, dict):
        log.warning(f"segments.build_video_segments_from_result invalid_result tool={tool_name!r} result_type={type(result).__name__!r}")
        err_result: Dict[str, Any] = {
            "meta": {"conversation_id": conversation_id, "trace_id": trace_id, "error": {"code": "invalid_result", "message": "result must be dict", "type": type(result).__name__}},
            "artifacts": [],
        }
        domain0 = "film2" if tool_name == "film2.run" else "video"
        segments_out = segments_out + [_base_segment(tool_name=tool_name, domain=domain0, trace_id=trace_id, conversation_id=conversation_id, index=0, result=err_result)]
        return segments_out
    # Film2 returns a rich segment tree (film -> scenes -> shots -> clips) under result.meta.segments.
    # Expand that into per-segment SegmentResults so QA/committee can evaluate each level independently.
    if tool_name == "film2.run":
        meta_obj = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        segs = meta_obj.get("segments") if isinstance(meta_obj.get("segments"), dict) else {}
        if segs:
            # Film segment
            film = segs.get("film") if isinstance(segs.get("film"), dict) else {}
            film_id = film.get("segment_id") if isinstance(film.get("segment_id"), str) else "film"
            film_meta = film.get("meta") if isinstance(film.get("meta"), dict) else {}
            film_result: Dict[str, Any] = {
                "meta": {
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "film_id": film_id,
                    "prompt": film_meta.get("prompt"),
                    "duration_s": film_meta.get("duration_s"),
                    "timecode": {"start_s": 0.0, "end_s": float(film_meta.get("duration_s") or 0.0)},
                    "locks": (meta_obj.get("locks") if isinstance(meta_obj.get("locks"), dict) else meta_obj.get("locks")),
                },
                "artifacts": film.get("artifacts") if isinstance(film.get("artifacts"), list) else (result.get("artifacts") if isinstance(result.get("artifacts"), list) else []),
            }
            seg_film = _base_segment(tool_name=tool_name, domain="film2", trace_id=trace_id, conversation_id=conversation_id, index=0, result=film_result)
            seg_film["film_id"] = str(film_id)
            seg_film["name"] = "film"
            segments_out = segments_out + [seg_film]

            # Scenes + shots
            scenes = segs.get("scenes") if isinstance(segs.get("scenes"), list) else []
            shots = segs.get("shots") if isinstance(segs.get("shots"), list) else []

            # Compute rough ordering/timing from shots list (best effort).
            shot_by_id: Dict[str, Dict[str, Any]] = {}
            for sh in shots:
                if isinstance(sh, dict) and isinstance(sh.get("segment_id"), str):
                    shot_by_id[str(sh.get("segment_id"))] = sh

            scene_index = 0
            for sc in scenes:
                if not isinstance(sc, dict):
                    continue
                scene_id = sc.get("segment_id")
                if not isinstance(scene_id, str) or not scene_id:
                    continue
                sc_meta = sc.get("meta") if isinstance(sc.get("meta"), dict) else {}
                # Aggregate duration from children shots when possible.
                dur_total = 0.0
                for child_id in (sc.get("children") or []):
                    if not isinstance(child_id, str):
                        continue
                    child = shot_by_id.get(child_id)
                    if isinstance(child, dict):
                        cmeta = child.get("meta") if isinstance(child.get("meta"), dict) else {}
                        if isinstance(cmeta.get("duration_s"), (int, float)):
                            dur_total += float(cmeta.get("duration_s") or 0.0)
                scene_result: Dict[str, Any] = {
                    "meta": {
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "film_id": film_id,
                        "scene_id": scene_id,
                        "prompt": sc_meta.get("prompt"),
                        "duration_s": dur_total if dur_total > 0.0 else None,
                        "timecode": {"start_s": 0.0, "end_s": dur_total},
                        "locks": (meta_obj.get("locks") if isinstance(meta_obj.get("locks"), dict) else meta_obj.get("locks")),
                    },
                    "artifacts": sc.get("artifacts") if isinstance(sc.get("artifacts"), list) else [],
                }
                seg_scene = _base_segment(tool_name=tool_name, domain="film2", trace_id=trace_id, conversation_id=conversation_id, index=(1 + scene_index), result=scene_result)
                seg_scene["film_id"] = str(film_id)
                seg_scene["scene_id"] = str(scene_id)
                seg_scene["name"] = f"scene:{scene_id}"
                segments_out = segments_out + [seg_scene]
                scene_index += 1

            shot_index = 0
            for sh in shots:
                if not isinstance(sh, dict):
                    continue
                shot_id = sh.get("segment_id")
                if not isinstance(shot_id, str) or not shot_id:
                    continue
                sh_meta = sh.get("meta") if isinstance(sh.get("meta"), dict) else {}
                dur_s = float(sh_meta.get("duration_s") or 0.0)
                shot_result: Dict[str, Any] = {
                    "meta": {
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "film_id": film_id,
                        "scene_id": sh_meta.get("scene_id"),
                        "shot_id": shot_id,
                        "prompt": sh_meta.get("prompt"),
                        "duration_s": dur_s if dur_s > 0.0 else None,
                        "timecode": {"start_s": 0.0, "end_s": dur_s},
                        "locks": sh_meta.get("locks"),
                    },
                    "artifacts": sh.get("artifacts") if isinstance(sh.get("artifacts"), list) else [],
                }
                seg_shot = _base_segment(tool_name=tool_name, domain="film2", trace_id=trace_id, conversation_id=conversation_id, index=(1000 + shot_index), result=shot_result)
                seg_shot["film_id"] = str(film_id)
                seg_shot["shot_id"] = str(shot_id)
                seg_shot["name"] = f"shot:{shot_id}"
                segments_out = segments_out + [seg_shot]
                shot_index += 1

            # Clips: NEVER set segment["id"] = clip_id. Keep canonical segment id and store clip_id separately.
            clips = segs.get("clips") if isinstance(segs.get("clips"), list) else []
            for clip_index, clip in enumerate(clips):
                if not isinstance(clip, dict):
                    continue
                clip_id = clip.get("clip_id")
                if not isinstance(clip_id, str) or not clip_id:
                    continue
                seg_result = clip.get("result") if isinstance(clip.get("result"), dict) else {}
                # Preserve clip-level meta/artifacts when present outside result.
                if "meta" not in seg_result and isinstance(clip.get("meta"), dict):
                    seg_result = dict(seg_result)
                    seg_result["meta"] = dict(clip.get("meta") or {})
                if "artifacts" not in seg_result and isinstance(clip.get("artifacts"), list):
                    seg_result = dict(seg_result)
                    seg_result["artifacts"] = list(clip.get("artifacts") or [])
                seg_clip = _base_segment(tool_name=tool_name, domain="video", trace_id=trace_id, conversation_id=conversation_id, index=(2000 + clip_index), result=seg_result)
                seg_clip["film_id"] = str(film_id)
                seg_clip["clip_id"] = str(clip_id)
                seg_clip["name"] = f"clip:{clip_id}"
                if isinstance(clip.get("qa"), dict):
                    seg_clip["qa"] = dict(clip.get("qa") or {})
                if isinstance(clip.get("locks"), dict):
                    seg_clip["locks"] = dict(clip.get("locks") or {})
                    meta_clip = seg_clip.get("meta")
                    if isinstance(meta_clip, dict):
                        meta_clip["locks"] = dict(clip.get("locks") or {})
                segments_out = segments_out + [seg_clip]

            if not segments_out:
                log.warning(f"segments.build_video_segments_from_result film2_empty_segments tool={tool_name!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
                segments_out = segments_out + [_base_segment(tool_name=tool_name, domain="film2", trace_id=trace_id, conversation_id=conversation_id, index=0, result=result)]
            return segments_out

    domain = "film2" if tool_name == "film2.run" else "video"
    segments_out: List[SegmentResult] = []
    segments_out = segments_out + [_base_segment(tool_name=tool_name, domain=domain, trace_id=trace_id, conversation_id=conversation_id, index=0, result=result)]
    return segments_out


def build_segments_for_tool(tool_name: str, *, trace_id: str, conversation_id: str | None, result: Dict[str, Any]) -> List[SegmentResult]:
    segments: List[SegmentResult] = []
    # Normalize conversation_id to empty string if None
    conversation_id = conversation_id or ""
    if not isinstance(tool_name, str):
        log.warning(f"segments.build_segments_for_tool invalid_tool_name tool_type={type(tool_name).__name__!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
        err_result: Dict[str, Any] = {
            "meta": {"conversation_id": conversation_id, "trace_id": trace_id, "error": {"code": "invalid_tool_name", "message": "tool_name must be str", "type": type(tool_name).__name__}},
            "artifacts": [],
        }
        segments = segments + [_base_segment(tool_name="unknown", domain="video", trace_id=trace_id, conversation_id=conversation_id, index=0, result=err_result)]
        return segments
    if not isinstance(result, dict):
        log.warning(f"segments.build_segments_for_tool invalid_result tool={tool_name!r} result_type={type(result).__name__!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
        err_result: Dict[str, Any] = {
            "meta": {"conversation_id": conversation_id, "trace_id": trace_id, "error": {"code": "invalid_result", "message": "result must be dict", "type": type(result).__name__}},
            "artifacts": [],
        }
        segments = segments + [_base_segment(tool_name=tool_name, domain="video", trace_id=trace_id, conversation_id=conversation_id, index=0, result=err_result)]
        return segments
    if tool_name.startswith("image."):
        segments = build_image_segments_from_result(tool_name=tool_name, trace_id=trace_id, conversation_id=conversation_id, result=result)
    elif tool_name.startswith("music."):
        segments = build_music_segments_from_result(tool_name=tool_name, trace_id=trace_id, conversation_id=conversation_id, result=result)
    elif tool_name == "tts.speak":
        segments = build_tts_segments_from_result(tool_name=tool_name, trace_id=trace_id, conversation_id=conversation_id, result=result)
    elif tool_name == "audio.sfx.compose":
        segments = build_sfx_segments_from_result(tool_name=tool_name, trace_id=trace_id, conversation_id=conversation_id, result=result)
    elif tool_name.startswith("video.") or tool_name == "film2.run":
        segments = build_video_segments_from_result(tool_name=tool_name, trace_id=trace_id, conversation_id=conversation_id, result=result)
    else:
        segments = build_video_segments_from_result(tool_name=tool_name, trace_id=trace_id, conversation_id=conversation_id, result=result)
    if not segments:
        log.warning(f"segments.build_segments_for_tool produced_no_segments tool={tool_name!r} trace_id={trace_id!r} conversation_id={conversation_id!r}")
        segments = segments + [
            _base_segment(
                tool_name=tool_name,
                domain="video",
                trace_id=trace_id,
                conversation_id=conversation_id,
                index=0,
                result={"meta": {"conversation_id": conversation_id, "trace_id": trace_id, "error": {"code": "no_segments", "message": "segment builder produced no segments"}}, "artifacts": []},
            )
        ]
    return segments


ALLOWED_PATCH_TOOLS = {"image.refine.segment", "video.refine.clip", "music.refine.window", "tts.refine.segment"}


def filter_patch_plan(patch_plan: List[Dict[str, Any]], segments: List[SegmentResult]) -> List[Dict[str, Any]]:
    """
    Filter patch_plan to only include steps that target valid segments.
    
    Note: Each item in patch_plan is a "step" (patch operation), not a segment.
    Each step has a "segment_id" field that identifies which segment it targets.
    """
    if not segments:
        log.debug(f"segments.filter_patch_plan: empty segments list patch_plan_count={len(patch_plan) if isinstance(patch_plan, list) else 0}")
    if not patch_plan:
        log.debug(f"segments.filter_patch_plan: empty patch_plan segments_count={len(segments) if isinstance(segments, list) else 0}")
    
    segment_ids = {segment.get("segment_id") for segment in segments if isinstance(segment, dict) and isinstance(segment.get("segment_id"), str)}
    filtered: List[Dict[str, Any]] = []
    skipped_count = 0
    for step in patch_plan or []:
        if not isinstance(step, dict):
            skipped_count += 1
            log.debug(f"segments.filter_patch_plan: skipping non-dict step type={type(step).__name__!r}")
            continue
        step_tool = (step.get("tool_name") or step.get("tool") or "").strip()
        if not step_tool or step_tool not in ALLOWED_PATCH_TOOLS:
            skipped_count += 1
            log.debug(f"segments.filter_patch_plan: skipping step with invalid tool={step_tool!r} allowed={ALLOWED_PATCH_TOOLS}")
            continue
        segment_id = step.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            skipped_count += 1
            log.debug(f"segments.filter_patch_plan: skipping step with invalid segment_id={segment_id!r} tool={step_tool!r}")
            continue
        if segment_id not in segment_ids:
            skipped_count += 1
            log.debug(f"segments.filter_patch_plan: skipping step with segment_id not found={segment_id!r} tool={step_tool!r} available_count={len(segment_ids)}")
            continue
        args = step.get("args")
        if args is None and ("arguments" in step):
            args = step.get("arguments")
        if isinstance(args, dict):
            args = dict(args)
        elif isinstance(args, str):
            parser = JSONParser()
            try:
                parsed = parser.parse(text=(args or ""), expected_schema={})
                args = dict(parsed) if isinstance(parsed, dict) else {"_raw": args}
            except Exception as ex:
                log.warning(f"segments.filter_patch_plan: JSONParser.parse failed for args string tool={step_tool!r} segment_id={segment_id!r} ex={ex!r}")
                args = {"_raw": args}
        elif args is None:
            args = {}
        else:
            args = {"_raw": args}
        filtered.append({"tool_name": step_tool, "tool": step_tool, "segment_id": segment_id, "args": args})
    
    if skipped_count > 0:
        log.debug(f"segments.filter_patch_plan: filtered patch_plan original_count={len(patch_plan) if isinstance(patch_plan, list) else 0} filtered_count={len(filtered)} skipped={skipped_count}")
    return filtered

def enrich_patch_plan_for_tts_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich tts.refine.segment steps with src/source audio + lock_bundle derived from TTS segments.
    """
    index_by_segment_id: Dict[str, SegmentResult] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_id = segment.get("segment_id")
        domain = segment.get("domain")
        if isinstance(segment_id, str) and segment_id and isinstance(domain, str) and domain == "tts":
            index_by_segment_id[segment_id] = segment
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            enriched.append(step)
            continue
        tool_name = (step.get("tool_name") or step.get("tool") or "").strip()
        segment_id = step.get("segment_id")
        args_obj = step.get("args")
        if isinstance(args_obj, dict):
            args_obj = dict(args_obj)
        elif isinstance(args_obj, str):
            parser = JSONParser()
            try:
                parsed = parser.parse(text=(args_obj or ""), expected_schema={})
                args_obj = dict(parsed) if isinstance(parsed, dict) else {"_raw": args_obj}
            except Exception as ex:
                log.warning(f"segments.enrich_patch_plan_for_tts_segments: JSONParser.parse failed for args_obj string tool={tool_name!r} segment_id={segment_id!r} ex={ex!r}")
                args_obj = {"_raw": args_obj}
        elif args_obj is None:
            args_obj = {}
        else:
            args_obj = {"_raw": args_obj}
        if tool_name != "tts.refine.segment":
            enriched.append(step)
            continue
        if not isinstance(segment_id, str) or not segment_id:
            enriched.append(step)
            continue
        segment = index_by_segment_id.get(segment_id)
        if not isinstance(segment, dict):
            log.debug(f"segments.enrich_patch_plan_for_tts_segments: segment not found segment_id={segment_id!r} tool={tool_name!r} available_count={len(index_by_segment_id)}")
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        if "segment_id" not in args:
            args["segment_id"] = segment_id
        seg_result = segment.get("result") if isinstance(segment.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        artifacts = seg_result.get("artifacts") if isinstance(seg_result.get("artifacts"), list) else []
        if "src" not in args and "source_audio" not in args:
            src: Optional[str] = None
            if artifacts:
                first_art = artifacts[0]
                if isinstance(first_art, dict):
                    path = first_art.get("path")
                    view_url = first_art.get("view_url")
                    if isinstance(path, str) and path.strip():
                        src = path
                    elif isinstance(view_url, str) and view_url.strip():
                        src = view_url
            if isinstance(src, str) and src:
                args["src"] = src
        if "lock_bundle" not in args:
            locks = meta.get("locks")
            if isinstance(locks, dict):
                inner = locks.get("bundle")
                args["lock_bundle"] = inner if isinstance(inner, dict) else locks
            elif isinstance(segment.get("locks"), dict):
                args["lock_bundle"] = segment.get("locks")
        if "conversation_id" not in args:
            if isinstance(segment.get("conversation_id"), str) and str(segment.get("conversation_id") or "").strip():
                args["conversation_id"] = str(segment.get("conversation_id") or "").strip()
        enriched.append({"tool_name": tool_name, "tool": tool_name, "segment_id": segment_id, "args": args})
    return enriched


def enrich_patch_plan_for_sfx_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich audio.sfx.compose patch steps (if ever used) with src + lock_bundle.
    """
    index_by_segment_id: Dict[str, SegmentResult] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_id = segment.get("segment_id")
        domain = segment.get("domain")
        if isinstance(segment_id, str) and segment_id and isinstance(domain, str) and domain == "sfx":
            index_by_segment_id[segment_id] = segment
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            enriched.append(step)
            continue
        tool_name = (step.get("tool_name") or step.get("tool") or "").strip()
        segment_id = step.get("segment_id")
        args_obj = step.get("args")
        if isinstance(args_obj, dict):
            args_obj = dict(args_obj)
        elif isinstance(args_obj, str):
            parser = JSONParser()
            try:
                parsed = parser.parse(text=(args_obj or ""), expected_schema={})
                args_obj = dict(parsed) if isinstance(parsed, dict) else {"_raw": args_obj}
            except Exception as ex:
                log.warning(f"segments.enrich_patch_plan_for_tts_segments: JSONParser.parse failed for args_obj string tool={tool_name!r} segment_id={segment_id!r} ex={ex!r}")
                args_obj = {"_raw": args_obj}
        elif args_obj is None:
            args_obj = {}
        else:
            args_obj = {"_raw": args_obj}
        if tool_name != "audio.sfx.compose":
            enriched.append(step)
            continue
        if not isinstance(segment_id, str) or not segment_id:
            enriched.append(step)
            continue
        segment = index_by_segment_id.get(segment_id)
        if not isinstance(segment, dict):
            log.debug(f"segments.enrich_patch_plan_for_sfx_segments: segment not found segment_id={segment_id!r} tool={tool_name!r} available_count={len(index_by_segment_id)}")
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        if "segment_id" not in args:
            args["segment_id"] = segment_id
        seg_result = segment.get("result") if isinstance(segment.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        if "lock_bundle" not in args and isinstance(meta.get("locks"), dict):
            locks = meta.get("locks")
            if isinstance(locks, dict):
                inner = locks.get("bundle")
                args["lock_bundle"] = inner if isinstance(inner, dict) else locks
        if "conversation_id" not in args:
            if isinstance(segment.get("conversation_id"), str) and str(segment.get("conversation_id") or "").strip():
                args["conversation_id"] = str(segment.get("conversation_id") or "").strip()
        enriched.append({"tool_name": tool_name, "tool": tool_name, "segment_id": segment_id, "args": args})
    return enriched


def enrich_patch_plan_for_image_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich image.refine.segment steps with segment_id, prompt, source_image,
    and lock_bundle derived from the corresponding image segments.
    """
    index_by_segment_id: Dict[str, SegmentResult] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_id = segment.get("segment_id")
        domain = segment.get("domain")
        if isinstance(segment_id, str) and segment_id and isinstance(domain, str) and domain == "image":
            index_by_segment_id[segment_id] = segment
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            continue
        tool_name = (step.get("tool_name") or step.get("tool") or "").strip()
        segment_id = step.get("segment_id")
        args_obj = step.get("args")
        if isinstance(args_obj, dict):
            args_obj = dict(args_obj)
        elif isinstance(args_obj, str):
            parser = JSONParser()
            try:
                parsed = parser.parse(text=(args_obj or ""), expected_schema={})
                args_obj = dict(parsed) if isinstance(parsed, dict) else {"_raw": args_obj}
            except Exception as ex:
                log.warning(f"segments.enrich_patch_plan_for_sfx_segments: JSONParser.parse failed for args_obj string tool={tool_name!r} segment_id={segment_id!r} ex={ex!r}")
                args_obj = {"_raw": args_obj}
        elif args_obj is None:
            args_obj = {}
        else:
            args_obj = {"_raw": args_obj}
        if tool_name != "audio.sfx.compose":
            enriched.append(step)
            continue
        if not isinstance(segment_id, str) or not segment_id:
            enriched.append(step)
            continue
        segment = index_by_segment_id.get(segment_id)
        if not isinstance(segment, dict):
            log.debug(f"segments.enrich_patch_plan_for_image_segments: segment not found segment_id={segment_id!r} tool={tool_name!r} available_count={len(index_by_segment_id)}")
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        if "segment_id" not in args:
            args["segment_id"] = segment_id
        seg_result = segment.get("result") if isinstance(segment.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        artifacts = seg_result.get("artifacts") if isinstance(seg_result.get("artifacts"), list) else []
        if "prompt" not in args:
            prompt = meta.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                args["prompt"] = prompt
        if "source_image" not in args:
            source_image: Optional[str] = None
            if artifacts:
                first_art = artifacts[0]
                if isinstance(first_art, dict):
                    path = first_art.get("path")
                    view_url = first_art.get("view_url")
                    if isinstance(path, str) and path.strip():
                        source_image = path
                    elif isinstance(view_url, str) and view_url.strip():
                        source_image = view_url
            if isinstance(source_image, str) and source_image:
                args["source_image"] = source_image
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
            elif isinstance(segment.get("locks"), dict):
                args["lock_bundle"] = segment.get("locks")
        # Ensure conversation_id is carried through so lock/state updates remain traceable.
        if "conversation_id" not in args:
            if isinstance(segment.get("conversation_id"), str) and str(segment.get("conversation_id") or "").strip():
                args["conversation_id"] = str(segment.get("conversation_id") or "").strip()
        enriched.append({"tool_name": tool_name, "tool": tool_name, "segment_id": segment_id, "args": args})
    return enriched


def enrich_patch_plan_for_video_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich video.refine.clip steps with src, timecode, prompt, and locks
    derived from the corresponding video clip segments.
    """
    index_by_segment_id: Dict[str, SegmentResult] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_id = segment.get("segment_id")
        domain = segment.get("domain")
        if isinstance(segment_id, str) and segment_id and isinstance(domain, str) and domain in ("video", "film2"):
            index_by_segment_id[segment_id] = segment
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            continue
        tool_name = (step.get("tool_name") or step.get("tool") or "").strip()
        segment_id = step.get("segment_id")
        args_obj = step.get("args")
        if isinstance(args_obj, dict):
            args_obj = dict(args_obj)
        elif isinstance(args_obj, str):
            parser = JSONParser()
            try:
                parsed = parser.parse(text=(args_obj or ""), expected_schema={})
                args_obj = dict(parsed) if isinstance(parsed, dict) else {"_raw": args_obj}
            except Exception as ex:
                log.warning(f"segments.enrich_patch_plan_for_video_segments: JSONParser.parse failed for args_obj string tool={tool_name!r} segment_id={segment_id!r} ex={ex!r}")
                args_obj = {"_raw": args_obj}
        elif args_obj is None:
            args_obj = {}
        else:
            args_obj = {"_raw": args_obj}
        if tool_name != "video.refine.clip":
            enriched.append(step)
            continue
        if not isinstance(segment_id, str) or not segment_id:
            enriched.append(step)
            continue
        segment = index_by_segment_id.get(segment_id)
        if not isinstance(segment, dict):
            log.debug(f"segments.enrich_patch_plan_for_video_segments: segment not found segment_id={segment_id!r} tool={tool_name!r} available_count={len(index_by_segment_id)}")
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        seg_result = segment.get("result") if isinstance(segment.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        artifacts = seg_result.get("artifacts") if isinstance(seg_result.get("artifacts"), list) else []
        if "src" not in args:
            src: Optional[str] = None
            if artifacts:
                first_art = artifacts[0]
                if isinstance(first_art, dict):
                    path = first_art.get("path")
                    view_url = first_art.get("view_url")
                    if isinstance(path, str) and path.strip():
                        src = path
                    elif isinstance(view_url, str) and view_url.strip():
                        src = view_url
            if isinstance(src, str) and src:
                args["src"] = src
        if "timecode" not in args:
            timecode = meta.get("timecode")
            if isinstance(timecode, dict):
                args["timecode"] = timecode
        if "prompt" not in args:
            prompt = meta.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                args["prompt"] = prompt
        if "locks" not in args:
            locks = meta.get("locks")
            if isinstance(locks, dict):
                args["locks"] = locks
        if "width" not in args:
            width = meta.get("width")
            if isinstance(width, (int, float)):
                args["width"] = int(width)
        if "height" not in args:
            height = meta.get("height")
            if isinstance(height, (int, float)):
                args["height"] = int(height)
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
        # Ensure conversation_id is carried through for traceability and any downstream stateful tools.
        if "conversation_id" not in args:
            if isinstance(segment.get("conversation_id"), str) and str(segment.get("conversation_id") or "").strip():
                args["conversation_id"] = str(segment.get("conversation_id") or "").strip()
        enriched.append({"tool_name": tool_name, "tool": tool_name, "segment_id": segment_id, "args": args})
    return enriched


def enrich_patch_plan_for_music_segments(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
) -> List[Dict[str, Any]]:
    """
    Enrich music.refine.window steps with window/audio context derived from
    the corresponding music segments.
    """
    index_by_segment_id: Dict[str, SegmentResult] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_id = segment.get("segment_id")
        domain = segment.get("domain")
        if isinstance(segment_id, str) and segment_id and isinstance(domain, str) and domain == "music":
            index_by_segment_id[segment_id] = segment
    enriched: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            enriched.append(step)
            continue
        tool_name = (step.get("tool_name") or step.get("tool") or "").strip()
        segment_id = step.get("segment_id")
        args_obj = step.get("args")
        if isinstance(args_obj, dict):
            args_obj = dict(args_obj)
        elif isinstance(args_obj, str):
            parser = JSONParser()
            try:
                parsed = parser.parse(text=(args_obj or ""), expected_schema={})
                args_obj = dict(parsed) if isinstance(parsed, dict) else {"_raw": args_obj}
            except Exception as ex:
                log.warning(f"segments.enrich_patch_plan_for_music_segments: JSONParser.parse failed for args_obj string tool={tool_name!r} segment_id={segment_id!r} ex={ex!r}")
                args_obj = {"_raw": args_obj}
        elif args_obj is None:
            args_obj = {}
        else:
            args_obj = {"_raw": args_obj}
        if tool_name != "music.refine.window":
            enriched.append(step)
            continue
        if not isinstance(segment_id, str) or not segment_id:
            enriched.append(step)
            continue
        segment = index_by_segment_id.get(segment_id)
        if not isinstance(segment, dict):
            log.debug(f"segments.enrich_patch_plan_for_music_segments: segment not found segment_id={segment_id!r} tool={tool_name!r} available_count={len(index_by_segment_id)}")
            enriched.append(step)
            continue
        args: Dict[str, Any] = dict(args_obj)
        seg_result = segment.get("result") if isinstance(segment.get("result"), dict) else {}
        meta = seg_result.get("meta") if isinstance(seg_result.get("meta"), dict) else {}
        artifacts = seg_result.get("artifacts") if isinstance(seg_result.get("artifacts"), list) else []
        if "segment_id" not in args:
            args["segment_id"] = segment_id
        window_id = meta.get("window_id")
        if isinstance(window_id, str) and window_id and "window_id" not in args:
            args["window_id"] = window_id
        if "src" not in args:
            src = None
            if artifacts:
                first_art = artifacts[0]
                if isinstance(first_art, dict):
                    path = first_art.get("path")
                    view_url = first_art.get("view_url")
                    if isinstance(path, str) and path.strip():
                        src = path
                    elif isinstance(view_url, str) and view_url.strip():
                        src = view_url
            if isinstance(src, str) and src:
                args["src"] = src
        if "lock_bundle" not in args:
            locks = segment.get("locks")
            if isinstance(locks, dict):
                args["lock_bundle"] = locks
        if "conversation_id" not in args:
            if isinstance(segment.get("conversation_id"), str) and str(segment.get("conversation_id") or "").strip():
                args["conversation_id"] = str(segment.get("conversation_id") or "").strip()
        enriched.append({"tool_name": tool_name, "tool": tool_name, "segment_id": segment_id, "args": args})
    return enriched


async def apply_patch_plan(
    patch_plan: List[Dict[str, Any]],
    segments: List[SegmentResult],
    tool_runner,
    trace_id: Optional[str],
    parent_tool: Optional[str],
) -> (List[SegmentResult], List[Dict[str, Any]]):
    """
    Apply patch_plan steps to segments.
    
    Note: Each item in patch_plan is a "step" (patch operation), not a segment.
    - step: A patch operation with "tool_name" (or "tool" for backward compatibility), "segment_id", and "args"
    - segment_id: Identifies which segment (media segment) the step targets
    - step_id: Executor-level identifier returned from tool execution (different from segment_id)
    """
    if not segments:
        log.warning(f"qa.segments.apply_patch_plan: empty segments list trace_id={trace_id!r} parent_tool={parent_tool!r}")
    if not patch_plan:
        log.debug(f"qa.segments.apply_patch_plan: empty patch_plan trace_id={trace_id!r} parent_tool={parent_tool!r} segments_count={len(segments)}")
    
    index_by_segment_id: Dict[str, int] = {}
    valid_segments_count = 0
    for idx, segment in enumerate(segments):
        if isinstance(segment, dict) and isinstance(segment.get("segment_id"), str):
            segment_id = segment["segment_id"]
            # If duplicate segment_id exists, log warning but use first occurrence
            if segment_id in index_by_segment_id:
                log.warning(f"qa.segments.apply_patch_plan: duplicate segment_id={segment_id!r} first_idx={index_by_segment_id[segment_id]} current_idx={idx} trace_id={trace_id!r}")
            else:
                index_by_segment_id[segment_id] = idx
                valid_segments_count += 1
        elif not isinstance(segment, dict):
            log.warning(f"qa.segments.apply_patch_plan: invalid segment type={type(segment).__name__!r} idx={idx} trace_id={trace_id!r}")
        elif not isinstance(segment.get("segment_id"), str):
            log.warning(f"qa.segments.apply_patch_plan: segment missing segment_id idx={idx} segment_keys={sorted(segment.keys()) if isinstance(segment, dict) else []} trace_id={trace_id!r}")
    
    if valid_segments_count == 0 and segments:
        log.warning(f"qa.segments.apply_patch_plan: no valid segments with segment_id trace_id={trace_id!r} segments_count={len(segments)}")
    
    # Deep copy segments to avoid modifying originals
    updated_segments: List[SegmentResult] = [copy.deepcopy(s) if isinstance(s, dict) else s for s in segments]
    patch_results: List[Dict[str, Any]] = []
    for step in patch_plan or []:
        if not isinstance(step, dict):
            log.warning(f"qa.segments.apply_patch_plan: invalid step type={type(step).__name__!r} trace_id={trace_id!r}")
            continue
        tool_name = step.get("tool_name") or step.get("tool")
        segment_id = step.get("segment_id")
        args = step.get("args")
        if not isinstance(tool_name, str) or not tool_name:
            log.warning(f"qa.segments.apply_patch_plan: missing or invalid tool_name step_keys={sorted(step.keys()) if isinstance(step, dict) else []} trace_id={trace_id!r}")
            continue
        if not isinstance(args, dict):
            log.warning(f"qa.segments.apply_patch_plan: missing or invalid args tool_name={tool_name!r} args_type={type(args).__name__!r} trace_id={trace_id!r}")
            continue
        seg_index = index_by_segment_id.get(segment_id) if isinstance(segment_id, str) else None
        target_segment: Optional[SegmentResult] = None
        if isinstance(seg_index, int) and 0 <= seg_index < len(updated_segments):
            target_segment = updated_segments[seg_index]
        elif segment_id is not None:
            log.warning(f"qa.segments.apply_patch_plan: segment_id not found tool_name={tool_name!r} segment_id={segment_id!r} trace_id={trace_id!r} available_segment_ids={sorted(list(index_by_segment_id.keys()))[:10]}")
        tool_call = {"tool_name": tool_name, "name": tool_name, "arguments": args}
        tool_env: Dict[str, Any] = {}
        try:
            if tool_runner is not None:
                tool_env = await tool_runner(tool_call)
                if not isinstance(tool_env, dict):
                    log.warning(f"qa.segments.apply_patch_plan: tool_runner returned non-dict type={type(tool_env).__name__!r} tool_name={tool_name!r} segment_id={segment_id!r} trace_id={trace_id!r}")
                    tool_env = {}
        except Exception as ex:
            log.warning(f"qa.segments.apply_patch_plan tool_runner exception tool={tool_name!r} segment_id={segment_id!r} ex={ex!r}", exc_info=True)
            tool_env = {"error": {"code": "tool_runner_exception", "message": str(ex)}}
        error_obj: Any = None
        res_obj: Dict[str, Any] = {}
        step_id: Optional[str] = None
        args_echo: Dict[str, Any] = dict(args)
        if isinstance(tool_env, dict):
            error_obj = tool_env.get("error")
            step_id = tool_env.get("step_id") if isinstance(tool_env.get("step_id"), str) else None
            args_in = tool_env.get("args")
            if isinstance(args_in, dict) and args_in:
                args_echo = dict(args_in)
            inner = tool_env.get("result")
            if isinstance(inner, dict):
                res_obj = inner
            elif inner is not None:
                log.warning(f"qa.segments.apply_patch_plan: tool_env.result is not a dict type={type(inner).__name__!r} tool_name={tool_name!r} segment_id={segment_id!r} trace_id={trace_id!r}")
        patch_entry: Dict[str, Any] = {
            "tool_name": str(tool_name),
            "result": res_obj,
            "args": args_echo,
            "error": error_obj,
            "step_id": step_id,
            "segment_id": segment_id,
            "tool_name": tool_name,
            "tool": tool_name,
            "parent_tool": parent_tool,
        }
        patch_results.append(patch_entry)
        if target_segment is not None and isinstance(tool_env, dict) and not bool(error_obj):
            if isinstance(res_obj, dict):
                target_segment["result"] = res_obj
                try:
                    seg_trace_id = target_segment.get("trace_id") if isinstance(target_segment.get("trace_id"), str) else trace_id
                    seg_conversation_id = target_segment.get("conversation_id") if isinstance(target_segment.get("conversation_id"), str) else None
                    seg_tool_name = target_segment.get("tool_name") if isinstance(target_segment.get("tool_name"), str) else (parent_tool or tool_name)
                    target_segment["artifacts"] = _extract_artifacts(res_obj, trace_id=seg_trace_id, conversation_id=seg_conversation_id, tool_name=seg_tool_name)
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
                except Exception as ex:
                    log.warning(f"qa.segments.apply_patch_plan: failed to hydrate derived segment fields tool_name={tool_name!r} segment_id={segment_id!r} trace_id={trace_id!r} ex={ex!r}", exc_info=True)
            else:
                log.warning(f"qa.segments.apply_patch_plan: cannot update target_segment with non-dict res_obj type={type(res_obj).__name__!r} tool_name={tool_name!r} segment_id={segment_id!r} trace_id={trace_id!r}")
            # Emit refine trace event for successful updates (only when res_obj is a dict)
            if isinstance(res_obj, dict) and target_segment is not None and isinstance(tool_env, dict) and not bool(error_obj):
                try:
                    if isinstance(trace_id, str) and trace_id.strip():
                        seg_domain = target_segment.get("domain")
                        seg_locks = target_segment.get("locks") if isinstance(target_segment.get("locks"), dict) else None
                        conversation_id = target_segment.get("conversation_id")
                        new_artifacts = res_obj.get("artifacts") if isinstance(res_obj.get("artifacts"), list) else []
                        artifact_path: Optional[str] = None
                        artifact_view: Optional[str] = None
                        if new_artifacts:
                            first_art = new_artifacts[0]
                            if isinstance(first_art, dict):
                                path = first_art.get("path")
                                view_url = first_art.get("view_url")
                                if isinstance(path, str) and path.strip():
                                    artifact_path = path
                                if isinstance(view_url, str) and view_url.strip():
                                    artifact_view = view_url
                        row: Dict[str, Any] = {
                            "trace_id": trace_id,
                            "event": "refine",
                            "parent_tool": parent_tool,
                            "patch_tool": tool_name,
                            "segment_id": segment_id,
                            "domain": seg_domain,
                            "conversation_id": conversation_id,
                            "args": args_echo,
                            "locks": seg_locks,
                            "result_meta": res_obj.get("meta") if isinstance(res_obj.get("meta"), dict) else {},
                        }
                        if artifact_path is not None:
                            row["artifact_path"] = artifact_path
                        if artifact_view is not None:
                            row["artifact_view_url"] = artifact_view
                        trace_event("qa.segments.refine", row)
                except Exception as ex:
                    log.warning(f"qa.segments.apply_patch_plan: failed to emit refine trace row tool_name={tool_name!r} segment_id={segment_id!r} trace_id={trace_id!r} ex={ex!r}", exc_info=True)
    
    # Log summary of patch execution
    successful_patches = sum(1 for pr in patch_results if isinstance(pr, dict) and not bool(pr.get("error")))
    failed_patches = len(patch_results) - successful_patches
    if patch_results:
        log.info(f"qa.segments.apply_patch_plan: completed trace_id={trace_id!r} parent_tool={parent_tool!r} total_steps={len(patch_results)} successful={successful_patches} failed={failed_patches}")
    elif patch_plan:
        log.warning(f"qa.segments.apply_patch_plan: no patch_results generated trace_id={trace_id!r} parent_tool={parent_tool!r} patch_plan_count={len(patch_plan)}")
    
    return updated_segments, patch_results



