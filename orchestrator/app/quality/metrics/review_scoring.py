from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ...analysis.media import analyze_image, analyze_audio


def score_review_image(*, path: str, prompt: str) -> Dict[str, Any]:
    ai = analyze_image(path, prompt=prompt)
    sem = ai.get("semantics") if isinstance(ai.get("semantics"), dict) else {}
    score_block = ai.get("score") if isinstance(ai.get("score"), dict) else {}
    return {
        "image": {
            "overall": float(score_block.get("overall") or 0.0),
            "semantic": float(score_block.get("semantic") or 0.0),
            "technical": float(score_block.get("technical") or 0.0),
            "clip": float(sem.get("clip_score") or 0.0),
        }
    }


def score_review_audio(*, path: str) -> Dict[str, Any]:
    aa = analyze_audio(path)
    return {"audio": {"lufs": aa.get("lufs"), "tempo_bpm": aa.get("tempo_bpm")}}


def score_review(kind: str, *, path: str, prompt: str) -> Dict[str, Any]:
    k = (kind or "").strip().lower()
    if k == "image":
        return score_review_image(path=path, prompt=prompt)
    return score_review_audio(path=path)


def pick_first_review_artifacts(artifacts: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Fast-path scorer selection for the legacy /v1/review/loop endpoint.
    Returns (first_image_path, first_audio_path).
    """
    img_path = None
    aud_path = None
    if not isinstance(artifacts, list):
        return None, None
    for a in artifacts:
        if not isinstance(a, dict):
            continue
        kind = a.get("kind") or ""
        path = a.get("path")
        if not isinstance(kind, str) or not isinstance(path, str):
            continue
        if img_path is None and kind.startswith("image"):
            img_path = path
        if aud_path is None and kind.startswith("audio"):
            aud_path = path
        if img_path is not None and aud_path is not None:
            break
    return img_path, aud_path




