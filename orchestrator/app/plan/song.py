'''
song.py: LLM-driven Song Graph planner.
'''


from __future__ import annotations

import os
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..json_parser import JSONParser
from ..tracing.runtime import trace_event
from ..committee_client import committee_ai_text, committee_jsonify

# Single logger per module (no custom logger names)
log = logging.getLogger(__name__)


# Expected Song Graph shape for coercion. This is intentionally minimal and
# forgiving; callers may extend the structure over time.
SONG_GRAPH_SCHEMA: Dict[str, Any] = {
    "global": {
        "bpm": int,
        "time_signature": str,
        "approx_length_s": int,
        "key": str,
        "style_tags": [str],
        "energy_curve": list,
        "emotion_curve": list,
    },
    "sections": [
        {
            "section_id": str,
            "name": str,
            "type": str,
            "t_start": float,
            "t_end": float,
            "bars": int,
            "key": str,
            "target_energy": float,
            "target_emotion": str,
            "is_chorus": bool,
            "reuse_from_section_id": str,
            # Optional multi-voice assignment for this section. Entries must
            # reference voice_id values from the top-level voices array, which
            # in turn map to concrete voice_lock_id entries.
            "voice_ids": [str],
        }
    ],
    "lyrics": {
        "sections": [
            {
                "section_id": str,
                "lines": [
                    {
                        "line_id": str,
                        "text": str,
                        "syllables": int,
                    }
                ],
            }
        ]
    },
    "voices": [
        {
            "voice_id": str,
            "role": str,
            "voice_lock_id": str,
            "tags": [str],
        }
    ],
    "instruments": [
        {
            "instrument_id": str,
            "role": str,
            "sections": [str],
            "tags": [str],
        }
    ],
    "motifs": [
        {
            "motif_id": str,
            "description": str,
            "sections": [str],
        }
    ],
}


def _schema_to_template(expected: Any) -> Any:
    """
    Convert an expected-schema shape (dict/list/type primitives) into a JSON-serializable
    template object. Kept local: committee_client is only allowed to export the committee
    call + jsonify functions.
    """
    if isinstance(expected, dict):
        return {k: _schema_to_template(v) for k, v in expected.items()}
    if isinstance(expected, list):
        if not expected:
            return []
        return [_schema_to_template(expected[0])]
    if isinstance(expected, type):
        if issubclass(expected, bool):
            return False
        if issubclass(expected, int):
            return 0
        if issubclass(expected, float):
            return 0.0
        if issubclass(expected, str):
            return ""
        if issubclass(expected, (list, tuple, set)):
            return []
        if issubclass(expected, dict):
            return {}
        return None
    return expected


def _fix_mojibake(s: str) -> str:
    """
    Lightweight normalization for common UTF-8 mojibake artifacts that leak
    through some LLM backends (e.g., â\x80\x99 for right single quote).

    This keeps semantics intact while cleaning up text for downstream TTS/lyrics.
    """
    if not isinstance(s, str) or not s:
        return s
    replacements = {
        "â\x80\x99": "'",   # smart apostrophe → straight quote
        "â\x80¦": "...",    # ellipsis
        "â\x80\x9c": "\"",  # left double quote
        "â\x80\x9d": "\"",  # right double quote
    }
    out = s
    for bad, good in replacements.items():
        if bad in out:
            out = out.replace(bad, good)
    return out


def _normalize_text_fields(obj: Any) -> Any:
    """
    Recursively normalize all string fields in a Song Graph object.
    """
    if isinstance(obj, dict):
        return {k: _normalize_text_fields(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_text_fields(obj=v) for v in obj]
    if isinstance(obj, str):
        return _fix_mojibake(s=obj)
    return obj


async def plan_song_graph(
    user_text: str,
    length_s: int,
    bpm: Optional[int],
    key: Optional[str],
    *,
    trace_id: str,
    music_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LLM-driven Song Graph planner.

    Targets a compact Song Graph JSON shape under {"song": SONG_GRAPH_SCHEMA}.
    """
    text = str(user_text or "").strip()
    t0 = time.perf_counter()
    approx_len = 30
    _ls = int(length_s) if isinstance(length_s, (int, float)) else 0
    approx_len = _ls if _ls > 0 else 30
    bpm_val = int(bpm) if isinstance(bpm, (int, float)) and bpm and bpm > 0 else 0
    key_txt = str(key or "").strip()

    schema_wrapper = {"song": SONG_GRAPH_SCHEMA}
    profile_txt = ""
    if isinstance(music_profile, dict) and music_profile:
        profile_txt = json.dumps(music_profile, ensure_ascii=False, default=str)
    prompt = (
        "You are MusicOps Song Planner. Plan a complete song structure before any audio is generated.\n"
        + "Return ONLY JSON matching this schema exactly:\n"
        + json.dumps(_schema_to_template(schema_wrapper), ensure_ascii=False)
        + "\n\n"
        + "User request (music): "
        + text
        + "\n"
        + f"Target length_s: {approx_len}\n"
        + f"Preferred bpm (0 = auto): {bpm_val}\n"
        + f"Preferred key (empty = auto): {key_txt}\n"
        + (("\nReference music profile (blend these characteristics):\n" + profile_txt + "\n") if profile_txt else "")
        + "CRITICAL: You MUST include ALL required fields: global, sections, lyrics, voices, instruments, and motifs.\n"
        + "The JSON MUST be complete and valid. Do NOT truncate or omit any fields.\n"
        + "Fill in global, sections, lyrics, voices, instruments, and motifs. "
        + "Use section_ids and motif_ids that are stable and reusable.\n"
        + "If a field should be empty, use an empty array [] or empty object {} as appropriate, but DO include the field.\n"
    )

    log.info(
        "plan_song_graph.start trace_id=%s text_len=%d length_s=%d bpm=%s key=%s has_profile=%s",
        trace_id,
        len(text),
        approx_len,
        bpm_val if bpm_val else None,
        key_txt if key_txt else None,
        bool(profile_txt),
    )
    # First, route song planner through the main committee to produce a Song Graph text.
    env = await committee_ai_text(
        messages=[
            {"role": "system", "content": "You are SongOps. Output ONLY JSON per the song schema."},
            {"role": "user", "content": prompt},
        ],
        trace_id=trace_id,
    )
    song: Dict[str, Any] = {}
    if not isinstance(env, dict) or not env.get("ok"):
        # Log committee failure instead of silently returning an empty song graph.
        log.error(
            "plan_song_graph committee failed (trace_id=%s): env=%r",
            trace_id,
            env,
        )
    else:
        res_env = env.get("result") or {}
        txt = res_env.get("text") or ""

        # Then, run the Song Graph text through committee.jsonify to enforce strict JSON.
        parsed_raw = await committee_jsonify(
            raw_text=txt or "{}",
            expected_schema=schema_wrapper,
            trace_id=trace_id,
            temperature=0.0,
        )
        # Final schema coercion pass.
        parser = JSONParser()
        parsed = parser.parse(parsed_raw if parsed_raw is not None else "{}", schema_wrapper)
        parsed = parsed if isinstance(parsed, dict) else {}
        song_obj = parsed.get("song")
        if isinstance(song_obj, dict):
            # Normalize textual fields (lyrics, section names, motif descriptions, etc.)
            # to clean up common UTF-8 mojibake before downstream tools consume them.
            song_obj = _normalize_text_fields(obj=song_obj)
            # Ensure all required fields exist with proper defaults
            if "global" not in song_obj or not isinstance(song_obj.get("global"), dict):
                song_obj["global"] = {}
            if "sections" not in song_obj or not isinstance(song_obj.get("sections"), list):
                song_obj["sections"] = []
            if "lyrics" not in song_obj or not isinstance(song_obj.get("lyrics"), dict):
                song_obj["lyrics"] = {"sections": []}
            if "voices" not in song_obj or not isinstance(song_obj.get("voices"), list):
                song_obj["voices"] = []
            if "instruments" not in song_obj or not isinstance(song_obj.get("instruments"), list):
                song_obj["instruments"] = []
            if "motifs" not in song_obj or not isinstance(song_obj.get("motifs"), list):
                song_obj["motifs"] = []
            # Ensure lyrics.sections exists
            lyrics_obj = song_obj.get("lyrics") if isinstance(song_obj.get("lyrics"), dict) else {}
            if "sections" not in lyrics_obj or not isinstance(lyrics_obj.get("sections"), list):
                lyrics_obj["sections"] = []
                song_obj["lyrics"] = lyrics_obj
            sections = song_obj.get("sections") if isinstance(song_obj.get("sections"), list) else []
            lyrics_sections = lyrics_obj.get("sections") if isinstance(lyrics_obj.get("sections"), list) else []
            motifs = song_obj.get("motifs") if isinstance(song_obj.get("motifs"), list) else []
            voices = song_obj.get("voices") if isinstance(song_obj.get("voices"), list) else []
            trace_event(
                "planner.song_graph.plan",
                {
                    "trace_id": trace_id,
                    "length_s": approx_len,
                    "bpm": bpm_val,
                    "key": key_txt,
                    "has_music_profile": bool(music_profile),
                    "sections_count": len(sections),
                    "lyrics_sections_count": len(lyrics_sections),
                    "motifs_count": len(motifs),
                    "voices_count": len(voices),
                },
            )
            song = song_obj
        else:
            log.warning(
                "plan_song_graph parsed missing song trace_id=%s parsed_keys=%s parser_errors=%s",
                trace_id,
                sorted(list(parsed.keys())) if isinstance(parsed, dict) else type(parsed).__name__,
                parser.errors[-5:] if parser.errors else [],
            )
            # Return a minimal valid song structure instead of empty dict
            song = {
                "global": {"bpm": bpm_val if bpm_val > 0 else 120, "time_signature": "4/4", "approx_length_s": approx_len, "key": key_txt if key_txt else "C Major", "style_tags": [], "energy_curve": [], "emotion_curve": []},
                "sections": [],
                "lyrics": {"sections": []},
                "voices": [],
                "instruments": [],
                "motifs": [],
            }
    log.info("plan_song_graph.done trace_id=%s ok=%s dur_ms=%d keys=%s", trace_id, bool(song), int((time.perf_counter() - t0) * 1000), sorted(list(song.keys())) if isinstance(song, dict) else type(song).__name__)
    return song



