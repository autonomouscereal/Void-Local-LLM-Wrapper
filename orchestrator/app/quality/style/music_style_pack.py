from __future__ import annotations

from typing import Any, Dict, List

from ..eval.music_eval import eval_technical
from ..embeddings.clap import embed_music_clap as _embed_music_clap, cos_sim as _cos_sim, MusicEvalError


def build_style_pack(ref_paths: List[str]) -> Dict[str, Any]:
    """
    Build a style pack from one or more reference songs.

    Uses canonical quality.eval.music_eval.eval_technical() metrics and optionally CLAP embeddings.
    """
    refs: List[Dict[str, Any]] = []
    embeds: List[List[float]] = []
    for p in ref_paths:
        if not isinstance(p, str) or not p:
            continue
        meta = eval_technical(p)
        ref_entry: Dict[str, Any] = {"path": p, "metrics": meta}
        v = _embed_music_clap(p)
        ref_entry["embed"] = v
        embeds.append(v)
        refs.append(ref_entry)
    if not embeds:
        return {
            "refs": [],
            "style_embed": [],
            "error": {"code": "ValidationError", "message": "build_style_pack: no valid reference embeddings"},
        }
    dim = len(embeds[0])
    acc = [0.0] * dim
    for v in embeds:
        if len(v) != dim:
            return {
                "refs": refs,
                "style_embed": [],
                "error": {"code": "ValidationError", "message": "build_style_pack: inconsistent embedding dimensions"},
            }
        for i, x in enumerate(v):
            acc[i] += float(x)
    n = float(len(embeds))
    style_embed: List[float] = [x / n for x in acc]
    tempos: List[float] = []
    keys: List[str] = []
    genres: List[str] = []
    emotions: List[str] = []
    for r in refs:
        m = r.get("metrics") or {}
        t = m.get("tempo_bpm")
        if isinstance(t, (int, float)):
            tempos.append(float(t))
        k = m.get("key")
        if isinstance(k, str) and k:
            keys.append(k)
        g = m.get("genre")
        if isinstance(g, str) and g:
            genres.append(g)
        e = m.get("emotion")
        if isinstance(e, str) and e:
            emotions.append(e)
    return {
        "refs": refs,
        "style_embed": style_embed,
        "tempo_bpm_mean": sum(tempos) / len(tempos) if tempos else None,
        "tempo_bpm_min": min(tempos) if tempos else None,
        "tempo_bpm_max": max(tempos) if tempos else None,
        "keys": list({k for k in keys}),
        "genres": list({g for g in genres}),
        "emotions": list({e for e in emotions}),
    }


def style_score_for_track(track_path: str, style_pack: Dict[str, Any]) -> float:
    if not isinstance(style_pack, dict):
        return 0.0
    style_embed = style_pack.get("style_embed")
    if not isinstance(style_embed, list) or not style_embed:
        return 0.0
    v = _embed_music_clap(track_path)
    sim = _cos_sim(style_embed, v)
    return 0.5 * (sim + 1.0)




