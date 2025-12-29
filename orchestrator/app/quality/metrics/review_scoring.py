from __future__ import annotations

import os
import logging
import cv2  # type: ignore
from typing import Any, Dict, List, Optional, Tuple

from ...analysis.media import analyze_image, analyze_audio

log = logging.getLogger(__name__)


def score_review_image(*, path: str, prompt: str) -> Dict[str, Any]:
    if not path or not isinstance(path, str):
        log.warning(f"review_scoring.score_review_image: invalid path={path!r} prompt_len={len(prompt) if isinstance(prompt, str) else 0}")
        return {"image": {"overall": 0.0, "semantic": 0.0, "technical": 0.0, "clip": 0.0}}
    try:
        ai = analyze_image(path, prompt=prompt)
        if not isinstance(ai, dict):
            log.warning(f"review_scoring.score_review_image: analyze_image returned non-dict type={type(ai).__name__!r} path={path!r}")
            return {"image": {"overall": 0.0, "semantic": 0.0, "technical": 0.0, "clip": 0.0}}
        sem = ai.get("semantics")
        score_block = ai.get("score")
        return {
            "image": {
                "overall": float((score_block or {}).get("overall") or 0.0),
                "semantic": float((score_block or {}).get("semantic") or 0.0),
                "technical": float((score_block or {}).get("technical") or 0.0),
                "clip": float((sem or {}).get("clip_score") or 0.0),
            }
        }
    except Exception as ex:
        log.warning(f"review_scoring.score_review_image: exception path={path!r} ex={ex!r}", exc_info=True)
        return {"image": {"overall": 0.0, "semantic": 0.0, "technical": 0.0, "clip": 0.0}}


def score_review_audio(*, path: str) -> Dict[str, Any]:
    if not path or not isinstance(path, str):
        log.warning(f"review_scoring.score_review_audio: invalid path={path!r}")
        return {"audio": {"lufs": None, "tempo_bpm": None}}
    try:
        aa = analyze_audio(path)
        if not isinstance(aa, dict):
            log.warning(f"review_scoring.score_review_audio: analyze_audio returned non-dict type={type(aa).__name__!r} path={path!r}")
            return {"audio": {"lufs": None, "tempo_bpm": None}}
        return {"audio": {"lufs": aa.get("lufs"), "tempo_bpm": aa.get("tempo_bpm")}}
    except Exception as ex:
        log.warning(f"review_scoring.score_review_audio: exception path={ex!r}", exc_info=True)
        return {"audio": {"lufs": None, "tempo_bpm": None}}


def _video_first_frame_to_image_path(video_path: str) -> Optional[str]:
    if not video_path:
        return None
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    # Write deterministically next to the video (no randomness, no hidden dirs).
    out = video_path + ".frame0.jpg"
    cv2.imwrite(out, frame)
    return out if os.path.exists(out) else None


def score_review_video(*, path: str, prompt: str) -> Dict[str, Any]:
    if not path or not isinstance(path, str):
        log.warning(f"review_scoring.score_review_video: invalid path={path!r} prompt_len={len(prompt) if isinstance(prompt, str) else 0}")
        return {"video": {"frame0_path": None}, "audio": {"lufs": None, "tempo_bpm": None}}
    scores: Dict[str, Any] = {}
    try:
        frame_path = _video_first_frame_to_image_path(path)
        if frame_path:
            frame_scores = score_review_image(path=frame_path, prompt=prompt)
            if isinstance(frame_scores, dict):
                scores.update(frame_scores)
            else:
                log.warning(f"review_scoring.score_review_video: score_review_image returned non-dict type={type(frame_scores).__name__!r} frame_path={frame_path!r}")
        else:
            log.debug(f"review_scoring.score_review_video: no frame extracted path={path!r}")
        # Also run audio analysis on the container path (analyze_audio handles decode).
        audio_scores = score_review_audio(path=path)
        if isinstance(audio_scores, dict):
            scores.update(audio_scores)
        else:
            log.warning(f"review_scoring.score_review_video: score_review_audio returned non-dict type={type(audio_scores).__name__!r} path={path!r}")
        scores["video"] = {"frame0_path": frame_path}
        return scores
    except Exception as ex:
        log.warning(f"review_scoring.score_review_video: exception path={path!r} ex={ex!r}", exc_info=True)
        return {"video": {"frame0_path": None}, "audio": {"lufs": None, "tempo_bpm": None}}


def score_review(kind: str, *, path: str, prompt: str) -> Dict[str, Any]:
    k = kind.lower()
    if k == "image":
        return score_review_image(path=path, prompt=prompt)
    if k in ("video", "film2"):
        return score_review_video(path=path, prompt=prompt)
    return score_review_audio(path=path)


def _artifact_family(kind: str) -> Optional[str]:
    k = kind.lower()
    if k.startswith("image"):
        return "image"
    if k.startswith("video") or k in ("film2", "mp4", "mov"):
        return "video"
    if k.startswith("audio") or k.startswith("music") or k.startswith("tts") or k in ("wav", "mp3", "m4a", "flac"):
        return "audio"
    return None


def _artifact_tags(a: Dict[str, Any]) -> List[str]:
    tags = list(a.get("tags") or [])
    tag_single = a.get("tag")
    if tag_single:
        tags.append(tag_single)
    summary = a.get("summary")
    if summary:
        tags.append(summary)
    return [str(t).lower() for t in tags]


def _artifact_review_count(a: Dict[str, Any]) -> int:
    meta = a.get("meta") or {}
    review = meta.get("review") or {}
    count = review.get("count") or meta.get("review_count") or 0
    count_int = int(count)
    if count_int >= 0:
        return count_int
    for t in _artifact_tags(a):
        if str(t).startswith("review_count:"):
            return int(str(t).split(":", 1)[1])
    return 0


def _artifact_created_ms(a: Dict[str, Any]) -> int:
    meta = a.get("meta") or {}
    created_ms = meta.get("created_ms") or 0
    created_ms_int = int(created_ms)
    if created_ms_int > 0:
        return created_ms_int
    created_at = meta.get("created_at") or 0
    created_at_int = int(created_at)
    if created_at_int > 0:
        return created_at_int * 1000
    for t in _artifact_tags(a):
        if str(t).startswith("created_ms:"):
            return int(str(t).split(":", 1)[1])
    return 0


def _artifact_priority(family: str, tags: List[str], kind: str, *, review_count: int) -> int:
    """
    Higher is better. This is intentionally simple and deterministic.
    """
    score = 0
    # Prefer unreviewed artifacts aggressively.
    if review_count <= 0:
        score += 1000
    else:
        score -= 25 * int(review_count)
    # Strong signals
    for t in tags:
        if t in ("best", "final", "hero", "primary"):
            score += 100
        if "final" in t or "best" in t or "hero" in t:
            score += 25
        if "window" in t or "stem" in t:
            score -= 10
        if "preview" in t or "thumb" in t:
            score -= 5
    k = (kind or "").lower()
    # Prefer canonical artifact refs when present
    if family == "audio":
        if k in ("audio-ref", "music-ref"):
            score += 10
        if k == "audio-window":
            score -= 5
    if family == "video":
        if "clip" in tags:
            score -= 2
    return score


def select_review_artifacts(artifacts: Any) -> Dict[str, Any]:
    """
    Deterministically select review targets from an artifacts list.

    Returns:
      {
        "selected": {"image": [path...], "video": [path...], "audio": [path...]},
        "debug": {...}
      }
    """
    selected: Dict[str, List[str]] = {"image": [], "video": [], "audio": []}
    if not isinstance(artifacts, list):
        artifacts = []
    debug: Dict[str, Any] = {"total": len(artifacts), "candidates": {"image": 0, "video": 0, "audio": 0}}
    candidates: List[Dict[str, Any]] = []
    skipped_invalid = 0
    for a in artifacts:
        if not isinstance(a, dict):
            skipped_invalid += 1
            log.debug(f"review_scoring.select_review_artifacts: skipping non-dict artifact type={type(a).__name__!r}")
            continue
        kind = a.get("kind")
        path = a.get("path")
        if not kind or not path:
            skipped_invalid += 1
            log.debug(f"review_scoring.select_review_artifacts: skipping artifact missing kind or path kind={kind!r} path={path!r}")
            continue
        family = _artifact_family(kind)
        if family is None:
            skipped_invalid += 1
            log.debug(f"review_scoring.select_review_artifacts: skipping artifact with unknown family kind={kind!r} path={path!r}")
            continue
        tags = _artifact_tags(a)
        review_count = _artifact_review_count(a)
        created_ms = _artifact_created_ms(a)
        prio = _artifact_priority(family, tags, kind, review_count=review_count)
        candidates.append(
            {"family": family, "kind": kind, "path": path, "prio": prio, "tags": tags, "review_count": review_count, "created_ms": created_ms}
        )

    # Deduplicate by (family,path)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in candidates:
        k = (c.get("family"), c.get("path"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)

    # Stable sort: unreviewed first, then prio desc, then newest, then path asc
    ranked: List[tuple[tuple[int, int, int, str], Dict[str, Any]]] = []
    for candidate_entry in uniq:
        review_count = int(candidate_entry.get("review_count") or 0)
        sort_key = (
            0 if review_count <= 0 else 1,
            -int(candidate_entry.get("prio") or 0),
            -int(candidate_entry.get("created_ms") or 0),
            str(candidate_entry.get("path") or ""),
        )
        ranked.append((sort_key, candidate_entry))
    ranked.sort()
    uniq = [pair[1] for pair in ranked]

    for family in ("video", "image", "audio"):
        fam = [c for c in uniq if c.get("family") == family]
        debug["candidates"][family] = len(fam)
        for c in fam:
            selected[family].append(str(c.get("path") or ""))
        selected[family] = [p for p in selected[family] if p]
    
    if skipped_invalid > 0:
        log.debug(f"review_scoring.select_review_artifacts: skipped {skipped_invalid} invalid artifacts total={len(artifacts)} candidates={len(candidates)}")

    return {"selected": selected, "debug": debug}




