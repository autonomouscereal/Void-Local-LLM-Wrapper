from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import json
import traceback

from ...analysis.media import analyze_audio, _load_audio  # type: ignore
from ...json_parser import JSONParser
from ...committee_client import committee_ai_text  # type: ignore
from ...committee_client import committee_jsonify  # type: ignore
from ..embeddings.clap import embed_music_clap as _embed_music_clap, cos_sim as _cos_sim, MusicEvalError
from void_quality.thresholds import get_music_acceptance_thresholds as _get_music_acceptance_thresholds
from void_envelopes import _build_success_envelope, _build_error_envelope
from ...tracing.runtime import trace_event

log = logging.getLogger(__name__)

MUSIC_HERO_QUALITY_MIN = 0.85
MUSIC_HERO_FIT_MIN = 0.85
_MUSIC_ACCEPT_THRESHOLDS: Dict[str, float] | None = None


def _map_technical_to_score(metrics: Dict[str, Any]):
    lufs = metrics.get("lufs")
    tempo = metrics.get("tempo_bpm")
    key = metrics.get("key")
    score_components: List[float] = []
    if isinstance(lufs, (int, float)):
        lufs_val = float(lufs)
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


def _compute_extra_technical(path: str):
    y, sr = _load_audio(path)
    if not y or not sr:
        return {"crest_factor": None, "rms": None, "dynamic_range": None}
    vals = [float(v) for v in y]
    rms = (sum(v * v for v in vals) / float(len(vals))) ** 0.5
    peak = max(abs(v) for v in vals)
    crest = float(peak / rms) if rms > 0.0 else None
    return {"crest_factor": crest, "rms": rms, "dynamic_range": None}


def eval_technical(track_path: str):
    base = analyze_audio(track_path)
    extra = _compute_extra_technical(track_path)
    out: Dict[str, Any] = {}
    out.update(base)
    out.update(extra)
    out["technical_quality_score"] = _map_technical_to_score(out)
    return out


def eval_motif_locks(track_path: str, *, tempo_target: Optional[float], key_target: Optional[str]):
    ainfo = analyze_audio(track_path)
    tempo_detected = ainfo.get("tempo_bpm")
    key_detected = ainfo.get("key")
    tempo_lock = None
    if isinstance(tempo_detected, (int, float)) and isinstance(tempo_target, (int, float)) and float(tempo_target) > 0.0:
        try:
            tempo_lock = 1.0 - (abs(float(tempo_detected) - float(tempo_target)) / float(tempo_target))
            tempo_lock = 0.0 if tempo_lock < 0.0 else 1.0 if tempo_lock > 1.0 else tempo_lock
        except Exception:
            tempo_lock = None
    key_lock = None
    if isinstance(key_target, str) and key_target.strip() and isinstance(key_detected, str) and key_detected.strip():
        key_lock = 1.0 if key_target.strip().lower() == key_detected.strip().lower() else 0.0
    motif_lock = None
    if isinstance(tempo_lock, (int, float)) and isinstance(key_lock, (int, float)):
        motif_lock = min(float(tempo_lock), float(key_lock))
    ainfo["tempo_lock"] = tempo_lock
    ainfo["key_lock"] = key_lock
    ainfo["lyrics_lock"] = None
    ainfo["motif_lock"] = motif_lock
    return ainfo


def eval_style(track_path: str, style_pack: Optional[Dict[str, Any]]):
    try:
        if style_pack is None:
            embed = _embed_music_clap(track_path)
            return {"style_score": 0.0, "track_embed": embed, "per_ref_scores": {}, "style_eval_ok": True}
        style_embed = style_pack.get("style_embed")
        if not isinstance(style_embed, list) or not style_embed:
            return {
                "style_score": 0.0,
                "track_embed": None,
                "per_ref_scores": {},
                "style_eval_ok": False,
                "reason": "malformed style_pack: missing style_embed",
                "stack": "".join(traceback.format_stack()),
            }
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
        return {"style_score": style_score, "track_embed": track_embed, "per_ref_scores": per_ref_scores, "style_eval_ok": True}
    except MusicEvalError as e:
        return {
            "style_score": 0.0,
            "track_embed": None,
            "per_ref_scores": {},
            "style_eval_ok": False,
            "reason": str(e),
            "stack": traceback.format_exc(),
        }


def _eval_structure_from_song_graph(track_path: str, song_graph: Dict[str, Any]):
    sections = song_graph.get("sections") if isinstance(song_graph.get("sections"), list) else []
    if not sections:
        return {"energy_alignment_score": 0.0, "repetition_score": 0.0, "transition_score": 0.0}
    y, sr = _load_audio(track_path)
    if not y or not sr:
        return {"energy_alignment_score": 0.0, "repetition_score": 0.0, "transition_score": 0.0}
    vals = [float(v) for v in y]
    n = len(vals)
    if n <= 0:
        return {"energy_alignment_score": 0.0, "repetition_score": 0.0, "transition_score": 0.0}
    sec_energies: List[float] = []
    targets: List[float] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        t_start = sec.get("t_start")
        t_end = sec.get("t_end")
        if not isinstance(t_start, (int, float)) or not isinstance(t_end, (int, float)) or t_end <= t_start:
            continue
        s_idx = int(max(0, min(int(t_start * sr), n - 1)))
        e_idx = int(max(s_idx + 1, min(int(t_end * sr), n)))
        slice_vals = vals[s_idx:e_idx]
        if not slice_vals:
            continue
        rms = (sum(v * v for v in slice_vals) / float(len(slice_vals))) ** 0.5
        sec_energies.append(rms)
        tgt = sec.get("target_energy")
        targets.append(float(tgt) if isinstance(tgt, (int, float)) else 0.5)
    if not sec_energies:
        return {"energy_alignment_score": 0.0, "repetition_score": 0.0, "transition_score": 0.0}
    max_energy = max(sec_energies)
    norm_energies = [e / max_energy for e in sec_energies] if max_energy > 0.0 else [0.0] * len(sec_energies)
    diffs = [abs(ne - tgt) for ne, tgt in zip(norm_energies, targets)]
    mean_diff = sum(diffs) / float(len(diffs))
    energy_alignment_score = 1.0 - mean_diff
    energy_alignment_score = 0.0 if energy_alignment_score < 0.0 else 1.0 if energy_alignment_score > 1.0 else energy_alignment_score
    mean_energy = sum(norm_energies) / float(len(norm_energies))
    var = sum((e - mean_energy) ** 2 for e in norm_energies) / float(len(norm_energies))
    if var < 0.01:
        repetition_score = 0.3
    elif var > 0.25:
        repetition_score = 0.6
    else:
        repetition_score = 1.0
    if len(norm_energies) > 1:
        jumps = [abs(norm_energies[i + 1] - norm_energies[i]) for i in range(len(norm_energies) - 1)]
        mean_jump = sum(jumps) / float(len(jumps))
        transition_score = 1.0 - mean_jump
        transition_score = 0.0 if transition_score < 0.0 else 1.0 if transition_score > 1.0 else transition_score
    else:
        transition_score = 1.0
    return {"energy_alignment_score": energy_alignment_score, "repetition_score": repetition_score, "transition_score": transition_score}


def eval_structure(track_path: str, song_graph: Dict[str, Any]):
    return _eval_structure_from_song_graph(track_path, song_graph or {})


def eval_emotion(track_path: str):
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
    return {"emotion_guess": emotion, "tempo_bpm": tempo, "valence": valence, "arousal": arousal}


def _build_music_eval_summary(all_axes: Dict[str, Any], film_context: Optional[Dict[str, Any]]):
    axes_slim: Dict[str, Any] = {}
    for key, val in (all_axes or {}).items():
        if not isinstance(val, dict):
            axes_slim[key] = val
            continue
        cur = dict(val)
        if key == "style":
            cur.pop("track_embed", None)
            cur.pop("style_embed", None)
            cur.pop("per_ref_scores", None)
            cur.pop("refs", None)
        axes_slim[key] = cur
    payload: Dict[str, Any] = {"axes": axes_slim, "film_context": film_context or {}}
    return json.dumps(payload, ensure_ascii=False)


async def _call_music_eval_committee(summary: str, trace_id: str = ""):
    # Use provided trace_id or fallback to "music_eval" for backwards compatibility
    effective_trace_id = trace_id if trace_id and trace_id.strip() else "music_eval"
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
        {"role": "user", "content": summary},
    ]
    schema = {"overall_quality_score": float, "fit_score": float, "originality_score": float, "cohesion_score": float, "issues": [str]}
    env = await committee_ai_text(messages=messages, trace_id=effective_trace_id)
    result_payload: Dict[str, Any] = {
        "overall_quality_score": 0.0,
        "fit_score": 0.0,
        "originality_score": 0.0,
        "cohesion_score": 0.0,
        "issues": ["music_eval_committee_error"],
    }
    if isinstance(env, dict) and env.get("ok"):
        res = env.get("result") or {}
        txt = res.get("text") or ""
        parsed = await committee_jsonify(raw_text=txt or "{}", expected_schema=schema, trace_id=effective_trace_id, temperature=0.0)
        parser = JSONParser()
        coerced = parser.parse(parsed if parsed is not None else "{}", schema)
        if isinstance(coerced, dict):
            result_payload = coerced
    return result_payload


async def eval_aesthetic(all_axes: Dict[str, Any], film_context: Optional[Dict[str, Any]], trace_id: str = ""):
    summary = _build_music_eval_summary(all_axes, film_context)
    return await _call_music_eval_committee(summary, trace_id=trace_id)


def _axes_are_effectively_empty(all_axes: Dict[str, Any]):
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


async def compute_music_eval(track_path: str, song_graph: Dict[str, Any], style_pack: Optional[Dict[str, Any]], film_context: Optional[Dict[str, Any]], trace_id: Optional[str] = None, conversation_id: Optional[str] = None):
    """
    Canonical music eval entrypoint.
    NEVER raises; always returns a structured envelope with full details.
    """
    
    conversation_id = conversation_id if isinstance(conversation_id, str) else ""
    try:
        trace_event("music.eval.start", {"trace_id": trace_id, "conversation_id": conversation_id, "track_path": track_path})
        technical = eval_technical(track_path)
        style = eval_style(track_path, style_pack)
        structure = eval_structure(track_path, song_graph)
        emotion = eval_emotion(track_path)
        all_axes = {"technical": technical, "style": style, "structure": structure, "emotion": emotion}
        if _axes_are_effectively_empty(all_axes):
            aesthetic = {
                "overall_quality_score": 0.0,
                "fit_score": 0.0,
                "originality_score": 0.0,
                "cohesion_score": 0.0,
                "issues": ["music.eval: skipped (no usable features)"],
            }
            trace_event("music.eval.skipped", {"trace_id": trace_id, "conversation_id": conversation_id, "track_path": track_path, "reason": "no_usable_features"})
        else:
            effective_trace_id = str(trace_id or "").strip()
            aesthetic = await eval_aesthetic(all_axes, film_context, trace_id=effective_trace_id)
        overall_quality = float(aesthetic.get("overall_quality_score") or 0.0)
        fit_score = float(aesthetic.get("fit_score") or 0.0)
        overall = {"overall_quality_score": overall_quality, "fit_score": fit_score}
        trace_event("music.eval.complete", {"trace_id": trace_id, "conversation_id": conversation_id, "track_path": track_path, "overall_quality": overall_quality, "fit_score": fit_score})
        return _build_success_envelope(
            result={"technical": technical, "style": style, "structure": structure, "emotion": emotion, "aesthetic": aesthetic, "overall": overall},
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    except Exception as ex:
        log.error(f"music.eval.exception trace_id={trace_id!r} conversation_id={conversation_id!r} track_path={track_path!r} ex={ex!r}", exc_info=True)
        trace_event("music.eval.exception", {"trace_id": trace_id, "conversation_id": conversation_id, "track_path": track_path, "exception": str(ex)})
        return _build_error_envelope(
            code="music_eval_exception",
            message=f"Exception while computing music eval: {ex}",
            trace_id=trace_id,
            conversation_id=conversation_id,
            status=500,
            details={"track_path": track_path, "stack": traceback.format_exc()},
        )


def get_music_acceptance_thresholds():
    global _MUSIC_ACCEPT_THRESHOLDS
    if _MUSIC_ACCEPT_THRESHOLDS is None:
        _MUSIC_ACCEPT_THRESHOLDS = dict(_get_music_acceptance_thresholds() or {})
    return _MUSIC_ACCEPT_THRESHOLDS


