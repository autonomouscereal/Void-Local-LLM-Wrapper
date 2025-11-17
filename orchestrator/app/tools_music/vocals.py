from __future__ import annotations

import os
import wave
from typing import Any, Dict, List

from .common import ensure_dir
from ..tools_tts.speak import run_tts_speak


def _collect_lyrics_for_section_and_voice(song_graph: Dict[str, Any], section_id: str, voice_id: str) -> List[str]:
    """
    Minimal lyrics collector: return all lyric lines for the given section,
    preferring lines that explicitly target the given voice_id when present.

    If no per-line voice information is available, all section lyrics are
    treated as applicable to all voices assigned to that section.
    """
    if not isinstance(song_graph, dict):
        return []
    lyrics_root = song_graph.get("lyrics") if isinstance(song_graph.get("lyrics"), dict) else {}
    sections = lyrics_root.get("sections") if isinstance(lyrics_root.get("sections"), list) else []
    texts: List[str] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        sid = sec.get("section_id")
        if not isinstance(sid, str) or sid != section_id:
            continue
        lines = sec.get("lines") if isinstance(sec.get("lines"), list) else []
        voice_specific: List[str] = []
        generic: List[str] = []
        for ln in lines:
            if not isinstance(ln, dict):
                continue
            txt = ln.get("text")
            if not (isinstance(txt, str) and txt.strip()):
                continue
            line_voice_id = ln.get("voice_id")
            if isinstance(line_voice_id, str) and line_voice_id.strip():
                if line_voice_id.strip() == voice_id:
                    voice_specific.append(txt.strip())
            else:
                generic.append(txt.strip())
        # Prefer voice-specific lines when any exist; otherwise fall back to generic.
        if voice_specific:
            texts.extend(voice_specific)
        else:
            texts.extend(generic)
    return texts


def plan_vocal_segments(song_graph: Dict[str, Any], windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a list of vocal segments based on Song Graph windows and voices.

    Each segment:
      {
        "voice_lock_id": str,
        "window_id": str,
        "start_s": float,
        "end_s": float,
        "text": str,
      }
    """
    segments: List[Dict[str, Any]] = []
    if not isinstance(song_graph, dict):
        return segments
    voices = song_graph.get("voices") if isinstance(song_graph.get("voices"), list) else []
    voice_id_to_lock: Dict[str, str] = {}
    for v in voices:
        if not isinstance(v, dict):
            continue
        vid = v.get("voice_id")
        vlock = v.get("voice_lock_id")
        if isinstance(vid, str) and vid and isinstance(vlock, str) and vlock:
            voice_id_to_lock[vid] = vlock
    for win in windows or []:
        if not isinstance(win, dict):
            continue
        section_id = win.get("section_id")
        if not isinstance(section_id, str) or not section_id:
            continue
        t_start = win.get("t_start")
        t_end = win.get("t_end")
        if not isinstance(t_start, (int, float)) or not isinstance(t_end, (int, float)):
            continue
        if float(t_end) <= float(t_start):
            continue
        duration = float(t_end) - float(t_start)
        voice_ids = win.get("voice_ids") if isinstance(win.get("voice_ids"), list) else []
        for voice_id in voice_ids:
            if not isinstance(voice_id, str) or not voice_id:
                continue
            voice_lock_id = voice_id_to_lock.get(voice_id)
            if not voice_lock_id:
                continue
            phrases = _collect_lyrics_for_section_and_voice(song_graph, section_id, voice_id)
            if not phrases:
                continue
            seg_duration = duration / float(len(phrases))
            for idx, text in enumerate(phrases):
                seg_start = float(t_start) + idx * seg_duration
                seg_end = seg_start + seg_duration
                segments.append(
                    {
                        "voice_lock_id": voice_lock_id,
                        "window_id": win.get("window_id"),
                        "start_s": seg_start,
                        "end_s": seg_end,
                        "text": text,
                    }
                )
    return segments


def render_vocal_stems_for_track(
    job: Dict[str, Any],
    song_graph: Dict[str, Any],
    windows: List[Dict[str, Any]],
    lock_bundle: Dict[str, Any],
    backing_path: str,
    cid: str,
    manifest: Dict[str, Any],
    tts_provider,
) -> Dict[str, Any]:
    """
    Render per-voice vocal stems over the given backing track using TTS→RVC→VocalFix.

    Returns a list of stems:
      {
        "path": str,
        "voice_lock_id": str,
        "start_s": float,
        "end_s": float,
      }
    """
    segments = plan_vocal_segments(song_graph, windows)
    stems: List[Dict[str, Any]] = []
    if not segments:
        return {"stems": stems}
    # Directory for vocal stems under the same CID as music.
    stem_dir = os.path.join("/workspace", "uploads", "artifacts", "music", cid, "stems")
    ensure_dir(stem_dir)
    for idx, seg in enumerate(segments):
        text = seg.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        voice_lock_id = seg.get("voice_lock_id")
        if not isinstance(voice_lock_id, str) or not voice_lock_id:
            continue
        start_s = float(seg.get("start_s") or 0.0)
        end_s = float(seg.get("end_s") or start_s)
        max_seconds = max(1, int(end_s - start_s)) if end_s > start_s else 4
        tts_job: Dict[str, Any] = {
            "text": text,
            "voice_lock_id": voice_lock_id,
            "sample_rate": int(job.get("tts_sample_rate") or 22050),
            "max_seconds": max_seconds,
            "seed": job.get("seed"),
            "lock_bundle": lock_bundle,
            "cid": cid,
        }
        env = run_tts_speak(tts_job, tts_provider, manifest)
        # run_tts_speak returns either an error envelope or a normal envelope.
        if not isinstance(env, dict):
            return {
                "error": {
                    "code": "tts_env_invalid",
                    "message": "run_tts_speak did not return a dict envelope",
                    "voice_lock_id": voice_lock_id,
                }
            }
        if env.get("error"):
            return {
                "error": {
                    "code": "tts_stem_error",
                    "message": f"tts.speak failed for voice_lock_id={voice_lock_id}",
                    "detail": env.get("error"),
                }
            }
        meta = env.get("meta") if isinstance(env.get("meta"), dict) else {}
        artifacts = env.get("artifacts") if isinstance(env.get("artifacts"), list) else []
        stem_path: str | None = None
        for art in artifacts:
            if not isinstance(art, dict):
                continue
            kind = art.get("kind")
            art_id = art.get("id")
            if isinstance(kind, str) and kind == "audio-ref" and isinstance(art_id, str) and art_id:
                # TTS artifacts live in /workspace/uploads/artifacts/audio/tts/{cid}
                stem_path = os.path.join(
                    "/workspace",
                    "uploads",
                    "artifacts",
                    "audio",
                    "tts",
                    meta.get("cid") or cid,
                    art_id,
                )
                break
        if not stem_path or not os.path.exists(stem_path):
            return {
                "error": {
                    "code": "tts_stem_missing",
                    "message": f"tts.speak produced no audio-ref artifact for voice_lock_id={voice_lock_id}",
                }
            }
        # Copy/normalize stem into the music CID stems directory for mixdown.
        with wave.open(stem_path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
            data = wf.readframes(nframes)
        # Always write stems as 16-bit PCM WAV, preserving sample rate/channels.
        stem_out = os.path.join(stem_dir, f"{voice_lock_id}_{seg.get('window_id')}_{idx}.wav")
        with wave.open(stem_out, "wb") as wf_out:
            wf_out.setnchannels(ch)
            wf_out.setsampwidth(sw)
            wf_out.setframerate(sr)
            wf_out.writeframes(data)
        stems.append(
            {
                "path": stem_out,
                "voice_lock_id": voice_lock_id,
                "start_s": start_s,
                "end_s": end_s,
            }
        )
    return {"stems": stems}
