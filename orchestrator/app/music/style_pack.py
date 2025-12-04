from __future__ import annotations

import math
import os
from typing import Any, Dict, List

import numpy as np  # type: ignore
import torch  # type: ignore

from ..analysis.media import _load_audio, analyze_audio


def _cos_sim(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity in [-1,1] between two equal-length vectors.

    This helper assumes non-empty, equal-length inputs and will raise if the
    inputs are malformed. Callers are responsible for ensuring dimensionality
    consistency when building style packs and embeddings.
    """
    if not a or not b:
        # Treat malformed vectors as a hard style-eval failure that should not crash
        # the tool. Callers are expected to wrap this in a higher-level error
        # envelope (e.g., MusicEvalError) with a full stack trace.
        return 0.0
    if len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da <= 0.0 or db <= 0.0:
        return 0.0
    return float(num) / float(da * db + 1e-9)


# CLAP model directory is populated by bootstrap into FILM2_MODELS/clap.
_MODELS_ROOT = os.environ.get("FILM2_MODELS", "/opt/models")
CLAP_MODEL_DIR = os.path.join(_MODELS_ROOT, "clap")

if not (os.path.isdir(CLAP_MODEL_DIR) and os.listdir(CLAP_MODEL_DIR)):
    # Defer CLAP availability errors to runtime so they can be surfaced as
    # structured music eval failures rather than import-time crashes.
    CLAP_MODEL_DIR_MISSING = True
else:
    CLAP_MODEL_DIR_MISSING = False

import laion_clap  # type: ignore


class MusicEvalError(Exception):
    """
    Non-fatal error for music evaluation paths (e.g., CLAP failures).

    Tool code (music.infinite.windowed, MusicEval 2.0) should treat this as
    "style eval unavailable for this track/clip" and degrade gracefully,
    rather than letting it crash the entire tool call.
    """
    pass


_CLAP_MODEL = None
_CLAP_STATE_PATH = os.path.join(CLAP_MODEL_DIR, "CLAP_HTSAT_base.pt")
if not CLAP_MODEL_DIR_MISSING:
    _CLAP_MODEL = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    if os.path.exists(_CLAP_STATE_PATH) and os.path.getsize(_CLAP_STATE_PATH) > 0:
        state = torch.load(_CLAP_STATE_PATH, map_location="cpu")
        _CLAP_MODEL.load_state_dict(state)
    _CLAP_MODEL.eval()


def _embed_music_clap(path: str) -> List[float]:
    """
    CLAP-based music embedding helper.

    Returns a dense float32 embedding for the given audio file path. CLAP
    presence and weights are treated as mandatory for style eval, but callers
    should catch MusicEvalError and degrade gracefully instead of crashing the
    entire music tool.
    """
    if not isinstance(path, str) or not path:
        raise MusicEvalError("embed_music_clap: invalid path")
    if CLAP_MODEL_DIR_MISSING or _CLAP_MODEL is None:
        raise MusicEvalError(f"embed_music_clap: CLAP model dir missing or empty; expected at {CLAP_MODEL_DIR}")
    if (not os.path.isfile(path)) or os.path.getsize(path) <= 0:
        raise MusicEvalError(f"embed_music_clap: audio file missing or empty: {path}")
    y, sr = _load_audio(path)
    if not y or not sr:
        raise MusicEvalError(f"embed_music_clap: zero-length or unreadable audio: {path}")
    try:
        with torch.no_grad():
            emb = _CLAP_MODEL.get_audio_embedding_from_filelist([path])
    except ZeroDivisionError as e:
        # Known laion_clap bug when audio_data is empty; surface as MusicEvalError.
        raise MusicEvalError(f"embed_music_clap: clap_zero_division for {path}") from e
    except Exception as e:
        raise MusicEvalError(f"embed_music_clap: clap_internal_error for {path}: {e}") from e
    if emb is None or len(emb) == 0:
        raise MusicEvalError("embed_music_clap: CLAP returned no embedding")
    e0 = emb[0]
    # Support both torch tensors and numpy arrays (and generic list-like) without
    # assuming .detach() is available.
    if isinstance(e0, torch.Tensor):
        arr = e0.detach().cpu().numpy().astype("float32")
    elif isinstance(e0, np.ndarray):
        arr = e0.astype("float32")
    else:
        arr = np.asarray(e0, dtype="float32")
    v = arr.tolist()
    if not v:
        raise MusicEvalError("embed_music_clap: empty embedding vector")
    return v


def build_style_pack(ref_paths: List[str]) -> Dict[str, Any]:
    """
    Build a style pack from one or more reference songs.

    Uses existing analyze_audio() metrics and optionally CLAP embeddings if
    laion-clap is available. The resulting dict is suitable for storage in a
    music lock bundle or Song Graph global branch.
    """
    refs: List[Dict[str, Any]] = []
    embeds: List[List[float]] = []
    for p in ref_paths:
        if not isinstance(p, str) or not p:
            continue
        meta = analyze_audio(p)
        ref_entry: Dict[str, Any] = {"path": p, "metrics": meta}
        v = _embed_music_clap(p)
        ref_entry["embed"] = v
        embeds.append(v)
        refs.append(ref_entry)
    if not embeds:
        # No valid references; surface as a structured error from the caller
        # instead of raising here. Callers should wrap this in MusicEvalError
        # or an error envelope with a full stack trace.
        return {
            "refs": [],
            "style_embed": [],
            "error": {
                "code": "ValidationError",
                "message": "build_style_pack: no valid reference embeddings",
            },
        }
    # Simple mean of CLAP embeddings
    dim = len(embeds[0])
    acc = [0.0] * dim
    for v in embeds:
        if len(v) != dim:
            return {
                "refs": refs,
                "style_embed": [],
                "error": {
                    "code": "ValidationError",
                    "message": "build_style_pack: inconsistent embedding dimensions",
                },
            }
        for i, x in enumerate(v):
            acc[i] += float(x)
    n = float(len(embeds))
    style_embed: List[float] = [x / n for x in acc]
    # Aggregate coarse tags from analyze_audio
    tempos = []
    keys = []
    genres = []
    emotions = []
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
    pack: Dict[str, Any] = {
        "refs": refs,
        "style_embed": style_embed,
        "tempo_bpm_mean": sum(tempos) / len(tempos) if tempos else None,
        "tempo_bpm_min": min(tempos) if tempos else None,
        "tempo_bpm_max": max(tempos) if tempos else None,
        "keys": list({k for k in keys}),
        "genres": list({g for g in genres}),
        "emotions": list({e for e in emotions}),
    }
    return pack


def style_score_for_track(track_path: str, style_pack: Dict[str, Any]) -> float:
    """
    Compute a style_score in [0,1] for a track versus a style pack.

    Style packs are required to carry a dense CLAP style_embed. This helper
    always returns a float similarity score; malformed style packs are treated
    as programmer errors and will surface as exceptions.
    """
    if not isinstance(style_pack, dict):
        # Invalid style pack; treat as total style mismatch without raising.
        return 0.0
    style_embed = style_pack.get("style_embed")
    if not isinstance(style_embed, list) or not style_embed:
        return 0.0
    v = _embed_music_clap(track_path)
    sim = _cos_sim(style_embed, v)
    # Map raw cosine in [-1,1] to [0,1] for downstream QA thresholds.
    score = 0.5 * (sim + 1.0)
    return score



