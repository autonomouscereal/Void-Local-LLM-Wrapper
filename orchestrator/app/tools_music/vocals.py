from __future__ import annotations

import os
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
import wave
import traceback
from typing import Any, Dict, List

from .common import ensure_dir
from ..tools_tts.speak import run_tts_speak
from void_artifacts import artifact_id_to_safe_filename


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
        sec_id = sec.get("section_id")
        if not isinstance(sec_id, str) or sec_id != section_id:
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
        "segment_id": str,
        "voice_id": str,
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
                segment_id = f"seg_{section_id}_{voice_id}_{idx:03d}"
                segments.append(
                    {
                        "segment_id": segment_id,
                        "voice_id": voice_id,
                        "voice_lock_id": voice_lock_id,
                        "window_id": win.get("window_id"),
                        "start_s": seg_start,
                        "end_s": seg_end,
                        "text": text,
                    }
                )
    return segments


async def render_vocal_stems_for_track(
    song_graph: Dict[str, Any],
    windows: List[Dict[str, Any]],
    lock_bundle: Dict[str, Any],
    backing_path: str,
    conversation_id: str,
    manifest: Dict[str, Any],
    tts_provider,
    seed: Any = None,
    trace_id: str = "",
    tts_sample_rate: int = 22050,
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
    # Directory for vocal stems under the same conversation_id bucket as music.
    outdir_key = conversation_id
    stem_dir = os.path.join(UPLOAD_DIR, "artifacts", "music", outdir_key, "stems")
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
            "voice_id": seg.get("voice_id"),
            "segment_id": seg.get("segment_id"),
            "sample_rate": int(tts_sample_rate),
            "max_seconds": max_seconds,
            "seed": seed,
            "lock_bundle": lock_bundle,
            "conversation_id": conversation_id,
        }
        env = await run_tts_speak(job=tts_job, provider=tts_provider, manifest=manifest, trace_id=trace_id, conversation_id=conversation_id)
        # run_tts_speak returns either an error envelope or a normal envelope.
        if not isinstance(env, dict):
            return {
                "error": {
                    "code": "InternalError",
                    "message": "run_tts_speak did not return a dict envelope",
                    "voice_lock_id": voice_lock_id,
                    "stack": traceback.format_stack(),
                }
            }
        if env.get("error"):
            # Propagate downstream error (with its own stack) and add our call-site stack.
            return {
                "error": {
                    "code": (env.get("error") or {}).get("code") if isinstance(env.get("error"), dict) else "InternalError",
                    "message": f"tts.speak failed for voice_lock_id={voice_lock_id}",
                    "detail": env.get("error"),
                    "stack": (env.get("error") or {}).get("stack") if isinstance(env.get("error"), dict) else traceback.format_stack(),
                }
            }
        meta = env.get("meta") if isinstance(env.get("meta"), dict) else {}
        artifacts = env.get("artifacts") if isinstance(env.get("artifacts"), list) else []
        stem_path: str | None = None
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            kind = artifact.get("kind")
            if isinstance(kind, str) and kind == "audio":
                # Use the actual path from the artifact, not artifact_id (which may contain colons/special chars)
                path_val = artifact.get("path")
                if isinstance(path_val, str) and path_val and os.path.exists(path_val):
                    stem_path = path_val
                    break
                # Fallback: try to construct path from artifact_id if path not available
                artifact_id = artifact.get("artifact_id")
                if isinstance(artifact_id, str) and artifact_id:
                    # TTS artifacts live in /workspace/uploads/artifacts/audio/tts/{conversation_id}
                    # Convert artifact_id to safe filename (matching how files are saved)
                    safe_filename = artifact_id_to_safe_filename(artifact_id, ".wav")
                    candidate_path = os.path.join(
                        "/workspace",
                        "uploads",
                        "artifacts",
                        "audio",
                        "tts",
                        meta.get("conversation_id") or conversation_id,
                        safe_filename,
                    )
                    if os.path.exists(candidate_path):
                        stem_path = candidate_path
                        break
        if not stem_path or not os.path.exists(stem_path):
            return {
                "error": {
                    "code": "InternalError",
                    "message": f"tts.speak produced no audio-ref artifact for voice_lock_id={voice_lock_id}",
                    "stack": traceback.format_stack(),
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
                "voice_id": seg.get("voice_id"),
                "segment_id": seg.get("segment_id"),
                "start_s": start_s,
                "end_s": end_s,
            }
        )
    return {"stems": stems}
