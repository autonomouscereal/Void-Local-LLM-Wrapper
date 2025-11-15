from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import httpx  # type: ignore

from ..json_parser import JSONParser
from ..pipeline.compression_orchestrator import co_pack, frames_to_string
from ..datasets.trace import append_sample as _trace_append


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


async def plan_song_graph(
    user_text: str,
    length_s: int,
    bpm: Optional[int],
    key: Optional[str],
    *,
    trace_id: Optional[str] = None,
    music_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LLM-driven Song Graph planner.

    Uses the same CO packing and JSONParser infrastructure as the main planner,
    but targets a compact Song Graph JSON shape under {"song": SONG_GRAPH_SCHEMA}.
    """
    text = str(user_text or "").strip()
    approx_len = int(length_s) if length_s and length_s > 0 else 30
    bpm_val = int(bpm) if isinstance(bpm, (int, float)) and bpm and bpm > 0 else 0
    key_txt = str(key or "").strip()

    co_env = {
        "schema_version": 1,
        "trace_id": str(trace_id or ""),
        "call_kind": "planner",
        "model_caps": {"num_ctx": 8192},
        "user_turn": {"role": "user", "content": text},
        "history": [],
        "attachments": [],
        "tool_memory": [],
        "rag_hints": [],
        "roe_incoming_instructions": [],
        "subject_canon": {},
        "percent_budget": {
            "icw_pct": [65, 70],
            "tools_pct": [18, 20],
            "roe_pct": [5, 10],
            "misc_pct": [3, 5],
            "buffer_pct": 5,
        },
        "sweep_plan": ["0-90", "30-120", "60-150+wrap"],
    }
    co_out = co_pack(co_env)
    frames_text = frames_to_string(co_out.get("frames") or [])

    schema_wrapper = {"song": SONG_GRAPH_SCHEMA}
    profile_txt = ""
    if isinstance(music_profile, dict) and music_profile:
        try:
            profile_txt = json.dumps(music_profile, ensure_ascii=False)
        except Exception:
            profile_txt = ""
    prompt = (
        frames_text
        + "\n\n"
        + "You are MusicOps Song Planner. Plan a complete song structure before any audio is generated.\n"
        + "Return ONLY JSON matching this schema exactly:\n"
        + json.dumps(schema_wrapper, ensure_ascii=False)
        + "\n\n"
        + "User request (music): "
        + text
        + "\n"
        + f"Target length_s: {approx_len}\n"
        + f"Preferred bpm (0 = auto): {bpm_val}\n"
        + f"Preferred key (empty = auto): {key_txt}\n"
        + (("\nReference music profile (blend these characteristics):\n" + profile_txt + "\n") if profile_txt else "")
        + "Fill in global, sections, lyrics, voices, instruments, and motifs. "
        + "Use section_ids and motif_ids that are stable and reusable.\n"
    )

    base = os.getenv("QWEN_BASE_URL", "http://ollama_qwen:11435").rstrip("/")
    model = os.getenv("QWEN_MODEL_ID", "qwen2.5:14b").strip() or "qwen2.5:14b"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(base + "/api/generate", json=payload)
    except Exception:
        return {}
    if resp.status_code < 200 or resp.status_code >= 300:
        return {}

    txt = (JSONParser().parse(resp.text, {"response": str}) or {}).get("response") or "{}"
    parsed = JSONParser().parse(txt, schema_wrapper)
    song = parsed.get("song")
    if isinstance(song, dict):
        sections = song.get("sections") if isinstance(song.get("sections"), list) else []
        lyrics_obj = song.get("lyrics") if isinstance(song.get("lyrics"), dict) else {}
        lyrics_sections = lyrics_obj.get("sections") if isinstance(lyrics_obj.get("sections"), list) else []
        motifs = song.get("motifs") if isinstance(song.get("motifs"), list) else []
        voices = song.get("voices") if isinstance(song.get("voices"), list) else []
        _trace_append(
            "music",
            {
                "event": "music.song_graph.plan",
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
        return song
    return {}



