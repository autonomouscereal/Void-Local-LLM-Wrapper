from __future__ import annotations

from typing import Any, Dict, Optional

from ...analysis.media import analyze_audio, analyze_image


def image_overall_score(path: str, *, prompt: Optional[str]) -> float:
    ai = analyze_image(path, prompt=prompt)
    score_block = ai.get("score") if isinstance(ai.get("score"), dict) else {}
    return float(score_block.get("overall") or 0.0)


def image_clip_score(path: str, *, prompt: Optional[str]) -> float:
    ai = analyze_image(path, prompt=prompt)
    sem = ai.get("semantics") if isinstance(ai.get("semantics"), dict) else {}
    return float(sem.get("clip_score") or 0.0)


def audio_spectral_flatness(path: str) -> float:
    a = analyze_audio(path)
    return float(a.get("spectral_flatness") or 0.0)


def audio_analysis(path: str) -> Dict[str, Any]:
    return analyze_audio(path)




