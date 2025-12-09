from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
import traceback

from ..analysis.media import analyze_audio, _load_audio  # type: ignore
from ..json_parser import JSONParser
from ..committee_client import committee_jsonify  # type: ignore
from .style_pack import _embed_music_clap, _cos_sim, MusicEvalError


MUSIC_HERO_QUALITY_MIN = 0.85
MUSIC_HERO_FIT_MIN = 0.85
_MUSIC_ACCEPT_THRESHOLDS: Dict[str, float] | None = None


def _map_technical_to_score(metrics: Dict[str, Any]) -> float:
    """
    Map low-level audio metrics into a coarse technical quality score in [0,1].
    """
    lufs = metrics.get("lufs")
    tempo = metrics.get("tempo_bpm")
    key = metrics.get("key")

    score_components: List[float] = []

    if isinstance(lufs, (int, float)):
        lufs_val = float(lufs)
        # Target window around -14 LUFS with soft tolerance.
        if -18.0 <= lufs_val <= -10.0:
            score_components.append(1.0)
        elif -24.0 <= lufs_val <= -6.0:
            score_components.append(0.7)
        else:
            score_components.append(0.3)

    if isinstance(tempo, (int, float)) and float(tempo) > 0.0:
        score_components.append(1.0)

    if isinstance(key, str) and key:
        score_components.append(1.0)

    if not score_components:
        return 0.0
    return float(sum(score_components) / float(len(score_components)))


def _compute_extra_technical(path: str) -> Dict[str, Any]:
    """
    Cheap RMS-based technical extras (crest factor, dynamic range proxy).
    """
    y, sr = _load_audio(path)
    if not y or not sr:
        return {
            "crest_factor": None,
            "rms": None,
            "dynamic_range": None,
        }
    vals = [float(v) for v in y]
    rms = (sum(v * v for v in vals) / float(len(vals))) ** 0.5
    peak = max(abs(v) for v in vals)
    crest = float(peak / rms) if rms > 0.0 else None
    # Dynamic range proxy: fixed placeholder until a fuller implementation is added.
    dyn = None
    return {
        "crest_factor": crest,
        "rms": rms,
        "dynamic_range": dyn,
    }


def eval_technical(track_path: str) -> Dict[str, Any]:
    base = analyze_audio(track_path)
    extra = _compute_extra_technical(track_path)
    out: Dict[str, Any] = {}
    out.update(base)
    out.update(extra)
    out["technical_quality_score"] = _map_technical_to_score(out)
    return out


def eval_style(track_path: str, style_pack: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Style-axis evaluation using CLAP embeddings.

    CLAP failures (e.g., zero-length audio, laion_clap internal ZeroDivisionError)
    are treated as non-fatal for the overall music tool: we surface them via
    style_eval_ok=False and a reason string instead of raising to FastAPI.
    """
    try:
        if style_pack is None:
            embed = _embed_music_clap(track_path)
            return {
                "style_score": 0.0,
                "track_embed": embed,
                "per_ref_scores": {},
                "style_eval_ok": True,
            }

        style_embed = style_pack.get("style_embed")
            # For malformed style packs, fail loudly so configuration bugs are
            # surfaced immediately instead of being silently tolerated.

        track_embed = _embed_music_clap(track_path)
        raw_sim = _cos_sim(style_embed, track_embed)
        style_score = 0.5 * (raw_sim + 1.0)

        per_ref_scores: Dict[str, float] = {}
        refs = style_pack.get("refs") or []
        if isinstance(refs, list):
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                ref_path = ref.get("path")
                ref_embed = ref.get("embed")
                if not isinstance(ref_path, str) or not isinstance(ref_embed, list):
                    continue
                sim = _cos_sim(ref_embed, track_embed)
                per_ref_scores[ref_path] = 0.5 * (sim + 1.0)

        return {
            "style_score": style_score,
            "track_embed": track_embed,
            "per_ref_scores": per_ref_scores,
            "style_eval_ok": True,
        }
    except MusicEvalError as e:
        # CLAP failed for this track; surface as non-fatal style-axis failure with stack.
        return {
            "style_score": 0.0,
            "track_embed": None,
            "per_ref_scores": {},
            "style_eval_ok": False,
            "reason": str(e),
            "stack": traceback.format_exc(),
        }


def _eval_structure_from_song_graph(track_path: str, song_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight structure evaluation based on Song Graph sections and
    approximate per-section energy.
    """
    sections = song_graph.get("sections") if isinstance(song_graph.get("sections"), list) else []
    if not sections:
        return {
            "energy_alignment_score": 0.0,
            "repetition_score": 0.0,
            "transition_score": 0.0,
        }

    y, sr = _load_audio(track_path)
    if not y or not sr:
        return {
            "energy_alignment_score": 0.0,
            "repetition_score": 0.0,
            "transition_score": 0.0,
        }

    vals = [float(v) for v in y]
    n = len(vals)
    if n <= 0:
        return {
            "energy_alignment_score": 0.0,
            "repetition_score": 0.0,
            "transition_score": 0.0,
        }

    sec_energies: List[float] = []
    targets: List[float] = []

    for sec in sections:
        if not isinstance(sec, dict):
            continue
        t_start = sec.get("t_start")
        t_end = sec.get("t_end")
        if not isinstance(t_start, (int, float)) or not isinstance(t_end, (int, float)):
            continue
        if t_end <= t_start:
            continue
        s_idx = int(max(0, min(int(t_start * sr), n - 1)))
        e_idx = int(max(s_idx + 1, min(int(t_end * sr), n)))
        slice_vals = vals[s_idx:e_idx]
        if not slice_vals:
            continue
        rms = (sum(v * v for v in slice_vals) / float(len(slice_vals))) ** 0.5
        sec_energies.append(rms)
        tgt = sec.get("target_energy")
        if isinstance(tgt, (int, float)):
            targets.append(float(tgt))
        else:
            targets.append(0.5)

    if not sec_energies:
        return {
            "energy_alignment_score": 0.0,
            "repetition_score": 0.0,
            "transition_score": 0.0,
        }

    max_energy = max(sec_energies)
    norm_energies = [e / max_energy for e in sec_energies] if max_energy > 0.0 else [0.0] * len(sec_energies)

    # Energy alignment: how close per-section energy is to target_energy.
    diffs = [abs(ne - tgt) for ne, tgt in zip(norm_energies, targets)]
    mean_diff = sum(diffs) / float(len(diffs))
    energy_alignment_score = 1.0 - mean_diff
    if energy_alignment_score < 0.0:
        energy_alignment_score = 0.0
    if energy_alignment_score > 1.0:
        energy_alignment_score = 1.0

    # Repetition: prefer some variation but not chaos.
    mean_energy = sum(norm_energies) / float(len(norm_energies))
    var = sum((e - mean_energy) ** 2 for e in norm_energies) / float(len(norm_energies))
    # Map variance into [0,1] with a soft range.
    if var < 0.01:
        repetition_score = 0.3
    elif var > 0.25:
        repetition_score = 0.6
    else:
        repetition_score = 1.0

    # Transition score: how smooth section energies are between neighbors.
    if len(norm_energies) > 1:
        jumps = [abs(norm_energies[i + 1] - norm_energies[i]) for i in range(len(norm_energies) - 1)]
        mean_jump = sum(jumps) / float(len(jumps))
        transition_score = 1.0 - mean_jump
        if transition_score < 0.0:
            transition_score = 0.0
        if transition_score > 1.0:
            transition_score = 1.0
    else:
        transition_score = 1.0

    return {
        "energy_alignment_score": energy_alignment_score,
        "repetition_score": repetition_score,
        "transition_score": transition_score,
    }


def eval_structure(track_path: str, song_graph: Dict[str, Any]) -> Dict[str, Any]:
    return _eval_structure_from_song_graph(track_path, song_graph or {})


def eval_emotion(track_path: str) -> Dict[str, Any]:
    base = analyze_audio(track_path)
    emotion = base.get("emotion")
    tempo = base.get("tempo_bpm")

    valence = 0.5
    arousal = 0.5

    if isinstance(tempo, (int, float)):
        t = float(tempo)
        if t > 150:
            arousal = 0.9
        elif t < 80:
            arousal = 0.3
        else:
            arousal = 0.6

    if isinstance(emotion, str):
        e = emotion.strip().lower()
        if e == "excited":
            valence = 0.8
        elif e == "calm":
            valence = 0.7
        elif e == "neutral":
            valence = 0.5
        else:
            valence = 0.5

    return {
        "emotion_guess": emotion,
        "tempo_bpm": tempo,
        "valence": valence,
        "arousal": arousal,
    }


def _build_music_eval_summary(all_axes: Dict[str, Any], film_context: Optional[Dict[str, Any]]) -> str:
    # Strip out heavy embedding vectors and non-essential blobs so that the
    # LLM-based MusicEval path only sees compact, scalar-style metrics rather
    # than raw embeddings intended for the lock/RAG systems.
    axes_slim: Dict[str, Any] = {}
    for key, val in (all_axes or {}).items():
        if not isinstance(val, dict):
            axes_slim[key] = val
            continue
        cur = dict(val)
        if key == "style":
            # Embeddings and large per-ref maps are for lock/RAG, not the LLM.
            cur.pop("track_embed", None)
            cur.pop("style_embed", None)
            cur.pop("per_ref_scores", None)
            cur.pop("refs", None)
        axes_slim[key] = cur

    payload: Dict[str, Any] = {
        "axes": axes_slim,
        "film_context": film_context or {},
    }
    # This summary is consumed by LLM-based committee paths and must be valid
    # JSON. Use the standard json module here instead of the JSONParser, which
    # is focused on parsing/repair rather than serialization.
    return json.dumps(payload, ensure_ascii=False)


async def _call_music_eval_committee(summary: str) -> Dict[str, Any]:
    """
    Use the existing committee infrastructure to obtain aesthetic judgements.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are MusicEval. You receive JSON with technical/style/structure/emotion metrics "
                "for a music segment or track, plus optional film context. "
                "You MUST ALWAYS respond with a single JSON object with EXACTLY these keys: "
                '{"overall_quality_score": float, "fit_score": float, "originality_score": float, '
                '"cohesion_score": float, "issues": [str]}.\n'
                "- You MUST ALWAYS respond with exactly one JSON object; any natural language, explanations, or "
                "formatting outside the JSON object is strictly forbidden.\n"
                "- Do NOT add any extra keys or remove/rename any keys.\n"
                "- You MUST assume the incoming JSON is valid enough for scoring; you are not allowed to refuse or "
                "claim that you cannot provide a meaningful response.\n"
                '- Phrases like \"I cannot provide a meaningful response\" or other refusals are ILLEGAL outputs.\n'
                "- If the input is missing, empty, or meaningless, you MUST STILL return this JSON with all scores 0.0 "
                "and a short, machine-readable explanation string in issues[0]."
            ),
        },
        {
            "role": "user",
            "content": summary,
        },
    ]
    schema = {
        "overall_quality_score": float,
        "fit_score": float,
        "originality_score": float,
        "cohesion_score": float,
        "issues": [str],
    }

    async def _run() -> Dict[str, Any]:
        # First, obtain the raw MusicEval text via the main committee path.
        from ..committee_client import CommitteeClient  # type: ignore

        client = CommitteeClient()
        env = await client.run(messages, trace_id="music_eval")
        result = env.get("result") or {}
        txt = result.get("text") or ""
        # Then, pass the raw text through committee_jsonify to enforce the schema.
        parsed = await committee_jsonify(
            txt or "{}",
            expected_schema=schema,
            trace_id="music_eval",
            temperature=0.0,
        )
        return parsed if isinstance(parsed, dict) else {}

    return await _run()


async def eval_aesthetic(all_axes: Dict[str, Any], film_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _build_music_eval_summary(all_axes, film_context)
    judge = await _call_music_eval_committee(summary)
    return judge


def _axes_are_effectively_empty(all_axes: Dict[str, Any]) -> bool:
    """
    Heuristic to detect when we have no meaningful features for MusicEval.

    When this returns True, we skip calling the committee and instead return
    a static "no data" evaluation.
    """
    tech = all_axes.get("technical") if isinstance(all_axes.get("technical"), dict) else {}
    style = all_axes.get("style") if isinstance(all_axes.get("style"), dict) else {}
    struct = all_axes.get("structure") if isinstance(all_axes.get("structure"), dict) else {}
    emo = all_axes.get("emotion") if isinstance(all_axes.get("emotion"), dict) else {}

    tq = tech.get("technical_quality_score")
    if isinstance(tq, (int, float)) and float(tq) != 0.0:
        return False
    track_embed = style.get("track_embed")
    if isinstance(track_embed, list) and track_embed:
        return False
    for k in ("energy_alignment_score", "repetition_score", "transition_score"):
        v = struct.get(k)
        if isinstance(v, (int, float)) and float(v) != 0.0:
            return False
    if emo.get("emotion_guess") or emo.get("genre") or emo.get("tempo_bpm"):
        return False
    return True


async def compute_music_eval(
    track_path: str,
    song_graph: Dict[str, Any],
    style_pack: Optional[Dict[str, Any]],
    film_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Multi-axis evaluation entrypoint for music clips and full tracks.

    Returns:
      {
        "technical": {...},
        "style": {...},
        "structure": {...},
        "emotion": {...},
        "aesthetic": {...},
        "overall": {
          "overall_quality_score": float,
          "fit_score": float,
        },
      }
    """
    technical = eval_technical(track_path)
    style = eval_style(track_path, style_pack)
    structure = eval_structure(track_path, song_graph)
    emotion = eval_emotion(track_path)

    all_axes = {
        "technical": technical,
        "style": style,
        "structure": structure,
        "emotion": emotion,
    }
    if _axes_are_effectively_empty(all_axes):
        aesthetic = {
            "overall_quality_score": 0.0,
            "fit_score": 0.0,
            "originality_score": 0.0,
            "cohesion_score": 0.0,
            "issues": ["music.eval: skipped (no usable features)"],
        }
    else:
        aesthetic = await eval_aesthetic(all_axes, film_context)

    overall_quality = float(aesthetic.get("overall_quality_score") or 0.0)
    fit_score = float(aesthetic.get("fit_score") or 0.0)

    overall = {
        "overall_quality_score": overall_quality,
        "fit_score": fit_score,
    }

    return {
        "technical": technical,
        "style": style,
        "structure": structure,
        "emotion": emotion,
        "aesthetic": aesthetic,
        "overall": overall,
    }


def get_music_acceptance_thresholds() -> Dict[str, float]:
    """
    Load music acceptance thresholds from review/acceptance_audio.json.

    This helper enforces that the config must exist and be well-formed; any
    failure here will surface as an exception instead of silently falling back.
    """
    global _MUSIC_ACCEPT_THRESHOLDS
    if _MUSIC_ACCEPT_THRESHOLDS is not None:
        return _MUSIC_ACCEPT_THRESHOLDS
    # acceptance_audio.json lives at orchestrator/app/review/acceptance_audio.json
    root_dir = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(root_dir, "review", "acceptance_audio.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        txt = f.read()
    parser = JSONParser()
    cfg = parser.parse_superset(txt or "{}", {"music": dict})["coerced"]
    music_cfg = cfg.get("music") if isinstance(cfg.get("music"), dict) else {}
    overall_min = float(music_cfg.get("overall_quality_min"))
    fit_min = float(music_cfg.get("fit_score_min"))
    _MUSIC_ACCEPT_THRESHOLDS = {
        "overall_quality_min": overall_min,
        "fit_score_min": fit_min,
    }
    return _MUSIC_ACCEPT_THRESHOLDS


