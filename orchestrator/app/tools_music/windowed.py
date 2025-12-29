from __future__ import annotations

import os
# All artifacts must live under the shared uploads volume so the UI can fetch them.
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
import wave
import logging
import time
import hashlib
from typing import Any, Dict, List, Tuple, Optional

from .common import now_ts, ensure_dir, sidecar, stamp_env, music_edge_defaults
from ..quality.style.music_style_pack import style_score_for_track  # type: ignore
from ..quality.eval.music_eval import compute_music_eval, eval_motif_locks  # type: ignore
from ..artifacts.manifest import add_manifest_row
from ..artifacts.index import add_artifact as _ctx_add
from ..tracing.training import append_training_sample
from ..datasets.stream import append_row as _ds_append_row
from void_envelopes import normalize_to_envelope, normalize_envelope, bump_envelope, assert_envelope
from ..tracing.runtime import trace_event
from void_artifacts import build_artifact, artifact_id_to_safe_filename, generate_artifact_id


log = logging.getLogger(__name__)


# Regeneration thresholds and limits for MusicEval-driven windowed composition.
QUALITY_THRESHOLD_TRACK = 0.7
FIT_THRESHOLD_TRACK = 0.7
WINDOW_QUALITY_THRESHOLD = 0.6
WINDOW_FIT_THRESHOLD = 0.6
MAX_REGEN_PASSES = 2

# Hard cap for per-window generation length passed to the backend music engine.
# The infinite/windowed tool is responsible for stitching longer tracks.
MAX_WINDOW_SECONDS = 20


async def _eval_window_clip(
    win: Dict[str, Any],
    clip_path: str,
    conversation_id: str,
    music_branch: Dict[str, Any],
    trace_id: str = "",
) -> Dict[str, Any]:
    """
    Evaluate a single window clip using MusicEval 2.0 and attach results to the window.
    """
    global_block = music_branch.get("global") if isinstance(music_branch.get("global"), dict) else {}
    sections = music_branch.get("sections") if isinstance(music_branch.get("sections"), list) else []
    sec_obj = None
    if isinstance(win.get("section_id"), str):
        for sec in sections:
            if isinstance(sec, dict) and sec.get("section_id") == win.get("section_id"):
                sec_obj = sec
                break
    local_song_graph: Dict[str, Any] = {
        "global": global_block,
        "sections": [sec_obj] if isinstance(sec_obj, dict) else [],
        "lyrics": {},
        "voices": music_branch.get("voices") or [],
        "instruments": music_branch.get("instruments") or [],
        "motifs": music_branch.get("motifs") or [],
    }
    style_pack = music_branch.get("style_pack") if isinstance(music_branch.get("style_pack"), dict) else None
    film_context: Dict[str, Any] = {}
    eval_out = await compute_music_eval(
        track_path=clip_path,
        song_graph=local_song_graph,
        style_pack=style_pack,
        film_context=film_context,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )
    eval_inner = eval_out.get("result") if isinstance(eval_out, dict) and isinstance(eval_out.get("result"), dict) else {}
    overall = eval_inner.get("overall") if isinstance(eval_inner.get("overall"), dict) else {}
    win["eval"] = eval_inner
    win["quality_score"] = overall.get("overall_quality_score")
    win["fit_score"] = overall.get("fit_score")
    append_training_sample(
        "music",
        {
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "tool": "music.infinite.windowed",
            "window_id": win.get("window_id"),
            "section_id": win.get("section_id"),
            "eval": eval_out,
            "path": clip_path,
        },
    )
    return eval_out


async def _eval_full_track(full_path: str, music_branch: Dict[str, Any], trace_id: Optional[str] = None, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate the stitched full track for multi-axis MusicEval.
    """
    style_pack = music_branch.get("style_pack") if isinstance(music_branch.get("style_pack"), dict) else None
    
    conversation_id = conversation_id if isinstance(conversation_id, str) else ""
    track_eval = await compute_music_eval(
        track_path=full_path,
        song_graph=music_branch,
        style_pack=style_pack,
        film_context={},
        trace_id=trace_id,
        conversation_id=conversation_id,
    )
    return track_eval


def _find_bad_windows(windows: List[Dict[str, Any]]) -> List[str]:
    """
    Return a list of window_ids that fall below per-window eval thresholds.
    """
    ids: List[str] = []
    for win in windows:
        if not isinstance(win, dict):
            continue
        q = win.get("quality_score")
        f = win.get("fit_score")
        if isinstance(q, (int, float)) and isinstance(f, (int, float)):
            if float(q) < WINDOW_QUALITY_THRESHOLD or float(f) < WINDOW_FIT_THRESHOLD:
                wid = win.get("window_id")
                if isinstance(wid, str) and wid:
                    ids.append(wid)
    return ids


async def _recompose_single_window(
    *,
    provider,
    manifest: Dict[str, Any],
    outdir_key: str,
    win: Dict[str, Any],
    music_branch: Dict[str, Any],
    global_block: Dict[str, Any],
    bpm: int,
    params: Dict[str, Any],
    trace_id: str,
    conversation_id: str,
    prompt: str,
    instrumental_only: bool,
    seed: Any,
) -> Tuple[bytes, int, int]:
    """
    Regenerate audio and metrics for a single window, returning raw PCM + sr + ch.
    """
    sections = music_branch.get("sections") if isinstance(music_branch.get("sections"), list) else []
    section_index: Dict[str, Dict[str, Any]] = {}
    for sec in sections:
        if isinstance(sec, dict) and isinstance(sec.get("section_id"), str):
            section_index[str(sec.get("section_id"))] = sec
    sr_local = int(params.get("sample_rate") or 44100)
    ch_local = int(params.get("channels") or 2)
    window_id = str(win.get("window_id") or "")
    if not window_id:
        return b"", sr_local, ch_local
    sec_obj = section_index.get(win.get("section_id")) if isinstance(win.get("section_id"), str) else None
    prompt = prompt or ""
    prompt_parts: List[str] = []
    if prompt:
        prompt_parts.append(prompt)
    sec_type = str(sec_obj.get("type") or "").lower() if isinstance(sec_obj, dict) else ""
    if sec_type:
        prompt_parts.append(f"{sec_type} section of the song")
    target_energy = sec_obj.get("target_energy") if isinstance(sec_obj, dict) else None
    if isinstance(target_energy, (int, float)):
        if target_energy <= 0.33:
            prompt_parts.append("low energy, more relaxed feel")
        elif target_energy >= 0.66:
            prompt_parts.append("high energy, more intense feel")
        else:
            prompt_parts.append("medium energy")
    target_emotion = sec_obj.get("target_emotion") if isinstance(sec_obj, dict) else None
    if isinstance(target_emotion, str) and target_emotion.strip():
        prompt_parts.append(f"emotion: {target_emotion.strip()}")
    style_tags: List[str] = []
    global_styles = global_block.get("style_tags")
    if isinstance(global_styles, list):
        style_tags = [s for s in global_styles if isinstance(s, str) and s.strip()]
    elif isinstance(global_block.get("genre_tags"), list):
        style_tags = [s for s in global_block.get("genre_tags") if isinstance(s, str) and s.strip()]
    if style_tags:
        prompt_parts.append("style: " + ", ".join(style_tags))
    if isinstance(sec_obj, dict):
        v_ids = sec_obj.get("voice_ids")
        if isinstance(v_ids, list):
            roles = []
            voices_list = music_branch.get("voices") or []
            if isinstance(voices_list, list):
                vid_to_role: Dict[str, str] = {}
                for v in voices_list:
                    if not isinstance(v, dict):
                        continue
                    vid = v.get("voice_id")
                    role = v.get("role")
                    if isinstance(vid, str) and vid and isinstance(role, str) and role.strip():
                        vid_to_role[vid] = role.strip()
                for vid in v_ids:
                    if isinstance(vid, str) and vid:
                        r = vid_to_role.get(vid)
                        roles.append(r or vid)
            if roles:
                prompt_parts.append("featured voices: " + ", ".join(roles))
            win["voice_ids"] = [str(vid) for vid in v_ids if isinstance(vid, str) and vid]
    if bool(instrumental_only):
        prompt_parts.append("instrumental only, no vocals")
    if sec_type == "chorus":
        prompt_parts.append("strong memorable hook, repeat the main motif")
    elif sec_type == "intro":
        prompt_parts.append("atmospheric build-up, introduce main motif softly")
    elif sec_type == "bridge":
        prompt_parts.append("contrast section, increase tension before final chorus")
    win_prompt = ". ".join([p for p in prompt_parts if p]).strip() or (prompt or "song section")
    raw_seconds = float((win.get("t_end") or 0.0) - (win.get("t_start") or 0.0))
    win_seconds = int(max(1, int(round(raw_seconds))))
    if win_seconds > MAX_WINDOW_SECONDS:
        win_seconds = MAX_WINDOW_SECONDS
    compose_args = {
        "prompt": win_prompt,
        "bpm": global_block.get("tempo_bpm") or bpm,
        "length_s": win_seconds,
        "sample_rate": params.get("sample_rate"),
        "channels": params.get("channels"),
        "music_lock": music_branch,
        "seed": seed,
        # Correlation IDs for backend logs
        "conversation_id": conversation_id,
        "trace_id": trace_id,
    }
    res = provider.compose(compose_args)
    wav_bytes = res.get("wav_bytes") or b""
    # Generate unique artifact_id BEFORE creating file, then use it for filename
    window_id_clip = win.get("window_id") or window_id
    clip_artifact_id = generate_artifact_id(
        trace_id=trace_id,
        tool_name="music.infinite.windowed.window",
        conversation_id=conversation_id,
        suffix_data=f"{window_id_clip}:{len(wav_bytes)}",
    )
    # Create safe filename from artifact_id (artifact_id is already sanitized, but use helper for consistency)
    safe_filename = artifact_id_to_safe_filename(clip_artifact_id, ".wav")
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "music", outdir_key)
    ensure_dir(outdir)
    clip_path = os.path.join(outdir, safe_filename)
    with wave.open(clip_path, "wb") as wf:
        wf.setnchannels(ch_local)
        wf.setsampwidth(2)
        wf.setframerate(sr_local)
        wf.writeframes(wav_bytes)
    # Store artifact_id in window dict for later use
    win["artifact_id"] = clip_artifact_id
    add_manifest_row(manifest, clip_path, step_id="music.infinite.windowed")
    tempo_target = None
    if isinstance(sec_obj, dict) and isinstance(sec_obj.get("tempo_bpm"), (int, float)):
        tempo_target = float(sec_obj.get("tempo_bpm"))
    elif isinstance(global_block.get("tempo_bpm"), (int, float)):
        tempo_target = float(global_block.get("tempo_bpm"))
    key_target = None
    if isinstance(sec_obj, dict) and isinstance(sec_obj.get("key"), str):
        key_target = sec_obj.get("key")
    elif isinstance(global_block.get("key"), str):
        key_target = global_block.get("key")
    ainfo = eval_motif_locks(clip_path, tempo_target=tempo_target, key_target=key_target)
    win["artifact_path"] = clip_path
    win["metrics"] = ainfo
    await _eval_window_clip(win, clip_path, outdir_key, music_branch, trace_id=trace_id)
    return wav_bytes, sr_local, ch_local


def _stitch_from_windows(
    windows: List[Dict[str, Any]],
    crossfade_frames: int,
) -> Tuple[bytes, int, int]:
    """
    Helper to rebuild PCM from existing windows on disk using timing order.
    """
    clips: List[Tuple[bytes, int, int]] = []
    time_sorted_windows: List[Tuple[float, Dict[str, Any]]] = []
    for win in windows:
        if not isinstance(win, dict):
            continue
        t_start = win.get("t_start")
        try:
            t = float(t_start) if isinstance(t_start, (int, float)) else 0.0
        except Exception as ex:
            # Keep windows sortable even on bad timestamps, but log the issue.
            log.warning(f"music.windowed: invalid t_start={t_start!r} for window_id={win.get('window_id')!r}: {ex!r}")
            t = 0.0
        time_sorted_windows.append((t, win))
    ranked_windows: List[Tuple[float, int, Dict[str, Any]]] = []
    for window_index, window_entry in enumerate(time_sorted_windows):
        ranked_windows.append((float(window_entry[0]), int(window_index), window_entry[1]))
    ranked_windows.sort()
    time_sorted_windows = [(ranked[0], ranked[2]) for ranked in ranked_windows]
    base_sr: Optional[int] = None
    base_ch: Optional[int] = None
    for _, win in time_sorted_windows:
        clip_path = win.get("artifact_path")
        if not (isinstance(clip_path, str) and clip_path and os.path.exists(clip_path)):
            continue
        try:
            with wave.open(clip_path, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
        except Exception as ex:
            log.error(f"music.windowed: failed to read clip_path={clip_path!r} for window_id={win.get('window_id')!r}: {ex!r}")
            continue
        if base_sr is None:
            base_sr = sr
        if base_ch is None:
            base_ch = ch
        if sr != base_sr or ch != base_ch:
            continue
        clips.append((frames, sr, ch))
    if not clips:
        return b"", 44100, 2
    stitched_pcm, stitched_sr, stitched_ch = _stitch_windows_pcm(clips, crossfade_frames)
    return stitched_pcm, stitched_sr, stitched_ch


async def _regenerate_bad_windows_only(
    provider,
    manifest: Dict[str, Any],
    outdir_key: str,
    windows: List[Dict[str, Any]],
    bad_window_ids: List[str],
    music_branch: Dict[str, Any],
    global_block: Dict[str, Any],
    bpm: int,
    params: Dict[str, Any],
    crossfade_frames: int,
    trace_id: str = "",
    conversation_id: str = "",
    prompt: str = "",
    instrumental_only: bool = False,
    seed: Any = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Recompose only windows listed in bad_window_ids, re-evaluate them, and stitch a new full track.
    """
    bad_set = {wid for wid in bad_window_ids if isinstance(wid, str) and wid}
    for win in windows:
        wid = win.get("window_id")
        if isinstance(wid, str) and wid in bad_set:
            await _recompose_single_window(
                provider=provider,
                manifest=manifest,
                outdir_key=outdir_key,
                win=win,
                music_branch=music_branch,
                global_block=global_block,
                bpm=bpm,
                params=params,
                trace_id=trace_id,
                conversation_id=conversation_id,
                prompt=prompt,
                instrumental_only=instrumental_only,
                seed=seed,
            )
    stitched_pcm, stitched_sr, stitched_ch = _stitch_from_windows(windows, crossfade_frames)
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "music", outdir_key)
    ensure_dir(outdir)
    stem = f"windowed_regen_{now_ts()}"
    full_path = os.path.join(outdir, f"{stem}.wav")
    with wave.open(full_path, "wb") as wf_full:
        wf_full.setnchannels(stitched_ch)
        wf_full.setsampwidth(2)
        wf_full.setframerate(stitched_sr)
        wf_full.writeframes(stitched_pcm)
    add_manifest_row(manifest, full_path, step_id="music.infinite.windowed.full")
    return windows, full_path


async def _regenerate_full_music(
    provider,
    manifest: Dict[str, Any],
    outdir_key: str,
    windows: List[Dict[str, Any]],
    music_branch: Dict[str, Any],
    global_block: Dict[str, Any],
    bpm: int,
    params: Dict[str, Any],
    crossfade_frames: int,
    trace_id: str = "",
    conversation_id: str = "",
    prompt: str = "",
    instrumental_only: bool = False,
    seed: Any = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Rebuild all windows with provider.compose, re-evaluate, and stitch into a fresh full track.
    """
    new_clips: List[Tuple[bytes, int, int, Dict[str, Any]]] = []
    for win in windows:
        if not isinstance(win, dict):
            continue
        wav_bytes, sr_local, ch_local = await _recompose_single_window(
            provider=provider,
            manifest=manifest,
            outdir_key=outdir_key,
            win=win,
            music_branch=music_branch,
            global_block=global_block,
            bpm=bpm,
            params=params,
            trace_id=trace_id,
            conversation_id=conversation_id,
            prompt=prompt,
            instrumental_only=instrumental_only,
            seed=seed,
        )
        new_clips.append((wav_bytes, sr_local, ch_local, win))
    clip_pcm_list: List[Tuple[bytes, int, int]] = []
    for pcm, sr, ch, _ in new_clips:
        clip_pcm_list.append((pcm, sr, ch))
    stitched_pcm, stitched_sr, stitched_ch = _stitch_windows_pcm(clip_pcm_list, crossfade_frames)
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "music", outdir_key)
    ensure_dir(outdir)
    stem = f"windowed_regen_full_{now_ts()}"
    full_path = os.path.join(outdir, f"{stem}.wav")
    with wave.open(full_path, "wb") as wf_full:
        wf_full.setnchannels(stitched_ch)
        wf_full.setsampwidth(2)
        wf_full.setframerate(stitched_sr)
        wf_full.writeframes(stitched_pcm)
    add_manifest_row(manifest, full_path, step_id="music.infinite.windowed.full")
    return windows, full_path


def _song_global_defaults(prompt: str, length_s: int, bpm: int) -> Dict[str, Any]:
    seconds = int(length_s) if length_s and length_s > 0 else 30
    bpm_int = int(bpm) if bpm and bpm > 0 else 120
    bars_total = max(8, int(round((seconds / 60.0) * bpm_int / 4.0)))
    return {
        "artifact_id": None,
        "reference_tracks": [],
        "text_prompt": prompt,
        "genre_tags": [],
        "mood_tags": [],
        "embeddings": {},
        "tempo_bpm": bpm_int,
        "time_signature": "4/4",
        "key": None,
        "scale": None,
        "swing": 0.0,
        "structure_summary": {
            "bars_total": bars_total,
            "sections_order": ["intro", "verse_1", "chorus_1", "outro"],
        },
        "energy_envelope": [],
        "loudness_envelope_lufs": [],
        "constraints": {
            "lock_mode": "guide",
            "lock_genre": False,
            "lock_tempo": True,
            "lock_key": False,
            "lock_energy_shape": False,
            "lock_loudness_shape": False,
        },
    }


def _build_song_sections(global_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    bars_total = int((global_block.get("structure_summary") or {}).get("bars_total") or 64)
    tempo_bpm = int(global_block.get("tempo_bpm") or 120)
    key = global_block.get("key") or "C minor"
    per_section = max(8, bars_total // 4)
    sections: List[Dict[str, Any]] = []
    bar_cursor = 1
    section_defs = [
        ("intro", "intro"),
        ("verse_1", "verse"),
        ("chorus_1", "chorus"),
        ("outro", "outro"),
    ]
    for idx, (sec_id, sec_type) in enumerate(section_defs):
        bar_start = bar_cursor
        bar_end = min(bars_total, bar_start + per_section - 1)
        time_start = (bar_start - 1) * (60.0 / max(1.0, float(tempo_bpm))) * 4.0
        time_end = bar_end * (60.0 / max(1.0, float(tempo_bpm))) * 4.0
        sections.append(
            {
                "section_id": sec_id,
                "type": sec_type,
                "order_index": idx,
                "bar_start": bar_start,
                "bar_end": bar_end,
                "time_start": time_start,
                "time_end": time_end,
                "tempo_bpm": tempo_bpm,
                "key": key,
                "energy_target": 0.5,
                "loudness_target_lufs": -14.0,
                "tension_target": 0.5,
                "target_voices": [],
                "target_instruments": [],
                "motif_ids": [],
                "lyrics": {
                    "text": "",
                    "lock_mode": "off",
                    "syllables_per_bar": [],
                },
                "instrument_overrides": {},
                "constraints": {
                    "lock_mode": "guide",
                    "lock_energy": False,
                    "lock_loudness": False,
                    "lock_section_type": True,
                },
            }
        )
        bar_cursor = bar_end + 1
        if bar_cursor > bars_total:
            break
    return sections


def build_music_window_plan(song_graph: Dict[str, Any], window_bars: int, overlap_bars: int) -> List[Dict[str, Any]]:
    music_branch = song_graph or {}
    global_block = music_branch.get("global") if isinstance(music_branch.get("global"), dict) else {}
    sections = music_branch.get("sections") if isinstance(music_branch.get("sections"), list) else []
    if not global_block:
        global_block = _song_global_defaults("", 30, 120)
    if not sections:
        sections = _build_song_sections(global_block)
    tempo_bpm = int(global_block.get("tempo_bpm") or 120)
    bars_per_sec = max(1e-6, (tempo_bpm / 60.0) / 4.0)
    step = max(1, int(window_bars))
    overlap = max(0, int(overlap_bars))
    windows: List[Dict[str, Any]] = []
    per_section_windows: Dict[str, List[Dict[str, Any]]] = {}
    for sec in sections:
        sec_id = str(sec.get("section_id") or "")
        if not sec_id:
            continue
        bar_start = int(sec.get("bar_start") or 1)
        bar_end = int(sec.get("bar_end") or bar_start)
        current_bar = bar_start
        index = 0
        while current_bar <= bar_end:
            win_bar_start = current_bar
            win_bar_end = min(bar_end, win_bar_start + step - 1)
            t_start = sec.get("time_start")
            if isinstance(t_start, (int, float)):
                base_t = float(t_start)
            else:
                base_t = (bar_start - 1) / bars_per_sec
            rel_bars = win_bar_start - bar_start
            win_t_start = base_t + rel_bars / bars_per_sec
            win_t_end = base_t + (win_bar_end - bar_start + 1) / bars_per_sec
            window_id = f"{sec_id}_w{index}"
            win_obj = {
                    "window_id": window_id,
                    "section_id": sec_id,
                    "is_chorus": bool(sec.get("type") == "chorus"),
                    "reuse_from_window_id": None,
                    "bars_start": win_bar_start,
                    "bars_end": win_bar_end,
                    "t_start": win_t_start,
                    "t_end": win_t_end,
                }
            windows.append(win_obj)
            per_section_windows.setdefault(sec_id, []).append(win_obj)
            if win_bar_end >= bar_end:
                break
            advance = step - overlap if step > overlap else 1
            current_bar = win_bar_start + advance
            index += 1
    # Chorus reuse: treat the first chorus section as canonical, and reuse its
    # windows for later chorus sections when possible.
    canonical_chorus: Optional[List[Dict[str, Any]]] = None
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        if sec.get("type") != "chorus":
            continue
        sec_id = str(sec.get("section_id") or "")
        if not sec_id:
            continue
        win_list = per_section_windows.get(sec_id) or []
        if not win_list:
            continue
        if canonical_chorus is None:
            canonical_chorus = win_list
            continue
        for idx, win_obj in enumerate(win_list):
            if canonical_chorus and idx < len(canonical_chorus):
                src_win = canonical_chorus[idx]
                src_id = src_win.get("window_id")
                if isinstance(src_id, str) and src_id:
                    win_obj["reuse_from_window_id"] = src_id
    return windows


def _stitch_windows_pcm(clips: List[Tuple[bytes, int, int]], crossfade_frames: int) -> Tuple[bytes, int, int]:
    if not clips:
        return b"", 44100, 2
    base_sr = clips[0][1]
    base_ch = clips[0][2]
    out = bytearray()
    for idx, (pcm, sr, ch) in enumerate(clips):
        if sr != base_sr or ch != base_ch:
            continue
        if idx == 0 or crossfade_frames <= 0:
            out.extend(pcm)
            continue
        fade_bytes = crossfade_frames * ch * 2
        prev_len = len(out)
        if prev_len < fade_bytes or len(pcm) < fade_bytes:
            out.extend(pcm)
            continue
        start_prev = prev_len - fade_bytes
        blended = bytearray(fade_bytes)
        for sample_byte_offset in range(0, fade_bytes, 2):
            pos_prev = start_prev + sample_byte_offset
            pos_new = sample_byte_offset
            s_prev = int.from_bytes(out[pos_prev:pos_prev + 2], "little", signed=True)
            s_new = int.from_bytes(pcm[pos_new:pos_new + 2], "little", signed=True)
            frac = sample_byte_offset / max(2, float(fade_bytes))
            mix = int(s_prev * (1.0 - frac) + s_new * frac)
            if mix < -32768:
                mix = -32768
            if mix > 32767:
                mix = 32767
            blended[pos_new:pos_new + 2] = int(mix).to_bytes(2, "little", signed=True)
        out[start_prev:] = blended
        out.extend(pcm[fade_bytes:])
    return bytes(out), base_sr, base_ch


async def run_music_infinite_windowed(*, provider, manifest: Dict[str, Any], conversation_id: str = "", trace_id: str = "", prompt: str = "", length_s: int = 60, bpm: int | None = None, key: str | None = None, window_bars: int = 8, overlap_bars: int = 1, mode: str = "start", lock_bundle: Dict[str, Any] | None = None, instrumental_only: bool = False, character_id: str | None = None, seed: Any = None, artifact_id: str | None = None, sample_rate: int | None = None, channels: int | None = None, edge: bool = False, **kwargs) -> Dict[str, Any]:
    t0_all = time.time()
    outdir_key = conversation_id
    if trace_id:
        trace_event("tool.music.infinite.windowed.start", {"trace_id": trace_id, "conversation_id": conversation_id, "prompt_length": len(prompt)})
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "music", outdir_key)
    ensure_dir(outdir)
    params = music_edge_defaults(
        {
            "bpm": bpm,
            "length_s": length_s,
            "sample_rate": sample_rate,
            "channels": channels,
        },
        edge=bool(edge),
    )
    mode = str(mode or "start").strip().lower()
    length_s = int(params.get("length_s") or 60)
    bpm = int(params.get("bpm") or 120)
    window_bars = int(window_bars or 8)
    overlap_bars = int(overlap_bars or 1)
    lock_bundle = lock_bundle if isinstance(lock_bundle, dict) else {}
    music_branch = lock_bundle.get("music") if isinstance(lock_bundle.get("music"), dict) else {}
    if not music_branch:
        music_branch = {
            "global": _song_global_defaults(prompt, length_s, bpm),
            "voices": [],
            "instruments": [],
            "motifs": [],
            "sections": [],
            "events": [],
        }
        lock_bundle["music"] = music_branch
    global_block = music_branch.get("global") if isinstance(music_branch.get("global"), dict) else {}
    if not global_block:
        global_block = _song_global_defaults(prompt, length_s, bpm)
        music_branch["global"] = global_block
    sections = music_branch.get("sections") if isinstance(music_branch.get("sections"), list) else []
    if not sections:
        music_branch["sections"] = _build_song_sections(global_block)
        sections = music_branch["sections"]
    windows = music_branch.get("windows") if isinstance(music_branch.get("windows"), list) else []
    # Provider response scratchpad for metadata. Must be initialized even if no windows
    # are generated/executed to avoid UnboundLocalError on envelope construction.
    res: Dict[str, Any] = {}

    log.info(
        f"music.windowed.start conversation_id={conversation_id!r} trace_id={trace_id!r} mode={mode!r} "
        f"length_s={length_s!r} bpm={bpm!r} window_bars={window_bars!r} overlap_bars={overlap_bars!r} "
        f"existing_windows={(len(windows) if isinstance(windows, list) else 0)!r}"
    )
    if trace_id:
        try:
            trace_event(
                "music.windowed.start",
                {
                    "trace_id": trace_id,
                    "conversation_id": conversation_id,
                    "mode": mode,
                    "length_s": length_s,
                    "bpm": bpm,
                    "window_bars": window_bars,
                    "overlap_bars": overlap_bars,
                    "prompt_preview": (prompt[:120] if isinstance(prompt, str) else ""),
                    "prompt_len": (len(prompt) if isinstance(prompt, str) else 0),
                },
            )
        except Exception:
            log.debug(f"music.windowed.trace_append_start_failed conversation_id={conversation_id!r} trace_id={trace_id!r}", exc_info=True)

    # Distillation-grade dataset row for the tool input (before any generation).
    try:
        _ds_append_row(
            "tool_event",
            {
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "tool": "music.infinite.windowed",
                "tags": ["tool:music.infinite.windowed", "phase:start"],
                "inputs": {
                    "prompt": prompt,
                    "mode": mode,
                    "length_s": length_s,
                    "bpm": bpm,
                    "window_bars": window_bars,
                    "overlap_bars": overlap_bars,
                    "seed": seed,
                },
                "locks": (lock_bundle if isinstance(lock_bundle, dict) else {}),
                "meta": {},
            },
        )
    except Exception:
        log.debug(f"music.windowed.dataset.append_start_failed conversation_id={conversation_id!r} trace_id={trace_id!r}", exc_info=True)
    if mode == "start":
        if not windows:
            windows = build_music_window_plan(music_branch, window_bars, overlap_bars)
            music_branch["windows"] = windows
    elif mode == "extend":
        # Extend existing windows based on additional length_s, using the last
        # known tempo and key from the Song Graph. This is a minimal extension
        # that appends loop-style windows at the tail.
        if windows:
            last_bars_end = 0
            for win in windows:
                try:
                    be = int(win.get("bars_end") or win.get("bar_end") or 0)
                    if be > last_bars_end:
                        last_bars_end = be
                except Exception:
                    continue
        else:
            windows = build_music_window_plan(music_branch, window_bars, overlap_bars)
            music_branch["windows"] = windows
            last_bars_end = 0
            for win in windows:
                try:
                    be = int(win.get("bars_end") or win.get("bar_end") or 0)
                    if be > last_bars_end:
                        last_bars_end = be
                except Exception:
                    continue
        tempo_bpm = int(global_block.get("tempo_bpm") or bpm)
        bars_per_sec = max(1e-6, (tempo_bpm / 60.0) / 4.0)
        extra_bars = int(max(1, round((length_s * bars_per_sec))))
        current_bar = last_bars_end + 1
        index_offset = len(windows)
        while extra_bars > 0:
            win_bar_start = current_bar
            win_bar_end = win_bar_start + window_bars - 1
            rel_bars = win_bar_start - 1
            win_t_start = rel_bars / bars_per_sec
            win_t_end = (win_bar_end) / bars_per_sec
            window_id = f"loop_w{index_offset}"
            win_obj = {
                "window_id": window_id,
                "section_id": "loop",
                "is_chorus": False,
                "reuse_from_window_id": None,
                "bars_start": win_bar_start,
                "bars_end": win_bar_end,
                "t_start": win_t_start,
                "t_end": win_t_end,
            }
            windows.append(win_obj)
            extra_bars -= window_bars
            current_bar = win_bar_end + 1
            index_offset += 1
        music_branch["windows"] = windows
    window_clips: List[Tuple[bytes, int, int, Dict[str, Any]]] = []
    # Map window_id -> window dict for quick reuse lookups
    window_index: Dict[str, Dict[str, Any]] = {}
    # Build quick index for sections by id for lock scoring.
    section_index: Dict[str, Dict[str, Any]] = {}
    for sec in sections:
        if isinstance(sec, dict) and isinstance(sec.get("section_id"), str):
            section_index[str(sec.get("section_id"))] = sec
    for win in windows:
        if not isinstance(win, dict):
            continue
        window_id = str(win.get("window_id") or "")
        if not window_id:
            continue
        window_index[window_id] = win
        sr_local = int(params.get("sample_rate") or 44100)
        ch_local = int(params.get("channels") or 2)
        reuse_from = win.get("reuse_from_window_id")
        if isinstance(reuse_from, str) and reuse_from:
            src_win = window_index.get(reuse_from)
            if src_win and isinstance(src_win.get("artifact_path"), str):
                win["artifact_path"] = src_win.get("artifact_path")
                win["metrics"] = src_win.get("metrics")
                log.debug(
                    f"music.windowed.window.reuse conversation_id={conversation_id!r} trace_id={trace_id!r} "
                    f"window_id={window_id!r} reuse_from={reuse_from!r} path={win.get('artifact_path')!r}"
                )
                window_clips.append((b"", sr_local, ch_local, win))
                continue
        existing_path = win.get("artifact_path")
        if isinstance(existing_path, str) and existing_path:
            # Existing window from a prior run; rely on disk contents and metrics.
            log.debug(
                f"music.windowed.window.skip_existing conversation_id={conversation_id!r} trace_id={trace_id!r} "
                f"window_id={window_id!r} path={existing_path!r}"
            )
            window_clips.append((b"", sr_local, ch_local, win))
            continue
        if trace_id:
            try:
                trace_event(
                    "music.windowed.window.start",
                    {
                        "trace_id": trace_id,
                        "conversation_id": conversation_id,
                        "window_id": window_id,
                        "section_id": win.get("section_id"),
                        "t_start": win.get("t_start"),
                        "t_end": win.get("t_end"),
                    },
                )
            except Exception:
                log.debug(
                    f"music.windowed.trace_event_failed event={'music.windowed.window.start'!r} conversation_id={conversation_id!r} "
                    f"trace_id={trace_id!r} window_id={window_id!r}",
                    exc_info=True,
                )
        # Build a section-aware prompt for this window.
        sec_obj = section_index.get(win.get("section_id")) if isinstance(win.get("section_id"), str) else None
        base_prompt = prompt or ""
        prompt_parts: List[str] = []
        if base_prompt:
            prompt_parts.append(base_prompt)
        sec_type = str(sec_obj.get("type") or "").lower() if isinstance(sec_obj, dict) else ""
        if sec_type:
            prompt_parts.append(f"{sec_type} section of the song")
        target_energy = sec_obj.get("target_energy") if isinstance(sec_obj, dict) else None
        if isinstance(target_energy, (int, float)):
            if target_energy <= 0.33:
                prompt_parts.append("low energy, more relaxed feel")
            elif target_energy >= 0.66:
                prompt_parts.append("high energy, more intense feel")
            else:
                prompt_parts.append("medium energy")
        target_emotion = sec_obj.get("target_emotion") if isinstance(sec_obj, dict) else None
        if isinstance(target_emotion, str) and target_emotion.strip():
            prompt_parts.append(f"emotion: {target_emotion.strip()}")
        style_tags: List[str] = []
        global_styles = global_block.get("style_tags")
        if isinstance(global_styles, list):
            style_tags = [s for s in global_styles if isinstance(s, str) and s.strip()]
        elif isinstance(global_block.get("genre_tags"), list):
            style_tags = [s for s in global_block.get("genre_tags") if isinstance(s, str) and s.strip()]
        if style_tags:
            prompt_parts.append("style: " + ", ".join(style_tags))
        # Optional multi-voice hint and instrumental-only toggle.
        if isinstance(sec_obj, dict):
            v_ids = sec_obj.get("voice_ids")
            if isinstance(v_ids, list):
                roles = []
                voices_list = music_branch.get("voices") or []
                if isinstance(voices_list, list):
                    # Build a quick index voice_id -> role for richer prompts.
                    vid_to_role = {}
                    for v in voices_list:
                        if not isinstance(v, dict):
                            continue
                        vid = v.get("voice_id")
                        role = v.get("role")
                        if isinstance(vid, str) and vid and isinstance(role, str) and role.strip():
                            vid_to_role[vid] = role.strip()
                    for vid in v_ids:
                        if isinstance(vid, str) and vid:
                            r = vid_to_role.get(vid)
                            roles.append(r or vid)
                if roles:
                    prompt_parts.append("featured voices: " + ", ".join(roles))
                # Also record which logical voices this window expects, so downstream
                # vocal pipelines can render stems per voice_lock_id.
                win["voice_ids"] = [str(vid) for vid in v_ids if isinstance(vid, str) and vid]
        if instrumental_only:
            prompt_parts.append("instrumental only, no vocals")
        # Section-type specific hints.
        if sec_type == "chorus":
            prompt_parts.append("strong memorable hook, repeat the main motif")
        elif sec_type == "intro":
            prompt_parts.append("atmospheric build-up, introduce main motif softly")
        elif sec_type == "bridge":
            prompt_parts.append("contrast section, increase tension before final chorus")
        win_prompt = ". ".join([p for p in prompt_parts if p]).strip() or (prompt or "song section")
        # Per-window duration passed to the backend is strictly capped; longer
        # tracks are achieved by stitching windows, not by a single huge call.
        raw_seconds = float((win.get("t_end") or 0.0) - (win.get("t_start") or 0.0))
        win_seconds = int(max(1, int(round(raw_seconds))))
        if win_seconds > MAX_WINDOW_SECONDS:
            win_seconds = MAX_WINDOW_SECONDS
        compose_args = {
            "prompt": win_prompt,
            "bpm": global_block.get("tempo_bpm") or bpm,
            "length_s": win_seconds,
            "sample_rate": params.get("sample_rate"),
            "channels": params.get("channels"),
            "music_lock": music_branch,
            "seed": seed,
            # Correlation IDs for backend logs
            "conversation_id": conversation_id,
            "trace_id": trace_id,
        }
        t0_call = time.time()
        res = provider.compose(compose_args)
        call_ms = int((time.time() - t0_call) * 1000.0)
        wav_bytes = res.get("wav_bytes") or b""
        log.info(
            f"music.windowed.window.composed conversation_id={conversation_id!r} trace_id={trace_id!r} window_id={window_id!r} "
            f"seconds={win_seconds!r} bytes={(len(wav_bytes) if isinstance(wav_bytes, (bytes, bytearray)) else 0)!r} "
            f"ms={call_ms!r} model={(res.get('model') if isinstance(res, dict) else None)!r}"
        )
        clip_stem = f"{window_id}_{now_ts()}"
        clip_path = os.path.join(outdir, f"{clip_stem}.wav")
        with wave.open(clip_path, "wb") as wf:
            wf.setnchannels(ch_local)
            wf.setsampwidth(2)
            wf.setframerate(sr_local)
            wf.writeframes(wav_bytes)
        add_manifest_row(manifest, clip_path, step_id="music.infinite.windowed")
        # Per-window lock scores based on Song Graph targets.
        # Tempo lock: closeness of detected tempo to section/global tempo.
        sec_obj = section_index.get(win.get("section_id")) if isinstance(win.get("section_id"), str) else None
        tempo_target = None
        if isinstance(sec_obj, dict) and isinstance(sec_obj.get("tempo_bpm"), (int, float)):
            tempo_target = float(sec_obj.get("tempo_bpm"))
        elif isinstance(global_block.get("tempo_bpm"), (int, float)):
            tempo_target = float(global_block.get("tempo_bpm"))
        key_target = None
        if isinstance(sec_obj, dict) and isinstance(sec_obj.get("key"), str):
            key_target = sec_obj.get("key")
        elif isinstance(global_block.get("key"), str):
            key_target = global_block.get("key")
        ainfo = eval_motif_locks(clip_path, tempo_target=tempo_target, key_target=key_target)
        win["artifact_path"] = clip_path
        win["metrics"] = ainfo
        # Per-window multi-axis evaluation (MusicEval 2.0).
        await _eval_window_clip(win, clip_path, conversation_id, music_branch, trace_id=trace_id)
        # Dataset row per window (distillation + debugging)
        try:
            _ds_append_row(
                "tool_event",
                {
                    "conversation_id": conversation_id,
                    "trace_id": trace_id,
                    "tool": "music.infinite.windowed",
                    "tags": ["tool:music.infinite.windowed", "phase:window", f"window:{window_id}"],
                    "inputs": {
                        "window_id": window_id,
                        "section_id": win.get("section_id"),
                        "prompt": win_prompt,
                        "seconds": win_seconds,
                        "bpm": compose_args.get("bpm"),
                        "seed": compose_args.get("seed"),
                    },
                    "outputs": {"path": clip_path, "audio_ref": clip_path},
                    "metrics": {
                        "provider_ms": call_ms,
                        "quality_score": win.get("quality_score"),
                        "fit_score": win.get("fit_score"),
                    },
                    "qa": (win.get("eval") if isinstance(win.get("eval"), dict) else {}),
                    "meta": {},
                },
            )
        except Exception:
            log.debug(
                f"music.windowed.dataset.append_window_failed conversation_id={conversation_id!r} trace_id={trace_id!r} window_id={window_id!r}",
                exc_info=True,
            )

        if trace_id:
            try:
                trace_event(
                    "music.windowed.window.finish",
                    {
                        "trace_id": trace_id,
                        "conversation_id": conversation_id,
                        "window_id": window_id,
                        "bytes": (len(wav_bytes) if isinstance(wav_bytes, (bytes, bytearray)) else 0),
                        "provider_ms": call_ms,
                        "quality_score": win.get("quality_score"),
                        "fit_score": win.get("fit_score"),
                        "artifact_path": clip_path,
                    },
                )
            except Exception:
                log.debug(
                    f"music.windowed.trace_event_failed event={'music.windowed.window.finish'!r} conversation_id={conversation_id!r} "
                    f"trace_id={trace_id!r} window_id={window_id!r}",
                    exc_info=True,
                )
        window_clips.append((wav_bytes, sr_local, ch_local, win))
    # For canonical windows we have PCM in memory; for reuse windows we read
    # their audio from disk before stitching.
    clip_pcm_list: List[Tuple[bytes, int, int]] = []
    for pcm, sr, ch, win in window_clips:
        data = pcm
        if not data:
            clip_path = win.get("artifact_path")
            if isinstance(clip_path, str) and os.path.exists(clip_path):
                with open(clip_path, "rb") as fh:
                    data = fh.read()
        clip_pcm_list.append((data, sr, ch))
    crossfade_frames = 2048  # Fixed crossfade frames
    stitched_pcm, stitched_sr, stitched_ch = _stitch_windows_pcm(
        clip_pcm_list,
        crossfade_frames,
    )
    # Generate unique artifact_id BEFORE creating file, then use it for filename
    if not artifact_id:
        artifact_id = generate_artifact_id(
            trace_id=trace_id,
            tool_name="music.infinite.windowed.full",
            conversation_id=conversation_id,
            suffix_data=f"full:{len(stitched_pcm)}",
        )
    # Create safe filename from artifact_id
    safe_filename = artifact_id_to_safe_filename(artifact_id, ".wav")
    full_path = os.path.join(outdir, safe_filename)
    with wave.open(full_path, "wb") as wf_full:
        wf_full.setnchannels(stitched_ch)
        wf_full.setsampwidth(2)
        wf_full.setframerate(stitched_sr)
        wf_full.writeframes(stitched_pcm)
    stem = os.path.splitext(safe_filename)[0]
    add_manifest_row(manifest, full_path, step_id="music.infinite.windowed.full")
    _ctx_add(outdir_key, "audio", full_path, None, None, ["music", "windowed"], {"trace_id": trace_id, "tool": "music.infinite.windowed"})
    # Artifact kinds are used by the UI/url collectors. For music, ensure kinds start with "music"
    # and include a stable /uploads URL so the UI never has to guess paths.
    rel_full = f"/uploads/artifacts/music/{outdir_key}/{os.path.basename(full_path)}"
    # Use "audio-ref" for compatibility with orchestrator collectors, and keep "music" tags for modality.
    artifacts: List[Dict[str, Any]] = [
        build_artifact(
            artifact_id=artifact_id,
            kind="audio",
            path=full_path,
            trace_id=trace_id,
            conversation_id=conversation_id,
            tool_name="music.infinite.windowed",
            summary=stem,
            bytes=len(stitched_pcm),
            url=rel_full,
            tags=[],
            window_id=window_id,
        )
    ]
    for _, _, _, win in window_clips:
        clip_path = win.get("artifact_path")
        if isinstance(clip_path, str):
            # Use artifact_id from window dict if available (generated when file was created)
            # Otherwise generate from artifact_id or create new one
            clip_artifact_id = win.get("artifact_id")
            if not clip_artifact_id:
                window_id_clip = win.get("window_id")
                if artifact_id and window_id_clip:
                    clip_artifact_id = f"{artifact_id}:window:{window_id_clip}"
                else:
                    file_size = os.path.getsize(clip_path) if os.path.exists(clip_path) else 0
                    clip_artifact_id = generate_artifact_id(
                        trace_id=trace_id,
                        tool_name="music.infinite.windowed.window",
                        conversation_id=conversation_id,
                        suffix_data=f"{window_id_clip}:{os.path.basename(clip_path)}:{file_size}",
                    )
            artifacts.append(
                build_artifact(
                    artifact_id=clip_artifact_id,
                    kind="audio-window",
                    path=clip_path,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    tool_name="music.infinite.windowed",
                    summary=win.get("window_id"),
                    bytes=os.path.getsize(clip_path) if os.path.exists(clip_path) else 0,
                    url=f"/uploads/artifacts/music/{outdir_key}/{os.path.basename(clip_path)}",
                    tags=[],
                    window_id=win.get("window_id"),
                )
            )
    # Aggregate basic audio metrics across windows for domain-level QA.
    lufs_vals: List[float] = []
    tempo_vals: List[float] = []
    key_vals: List[str] = []
    for _, _, _, win in window_clips:
        metrics = win.get("metrics") if isinstance(win.get("metrics"), dict) else {}
        lv = metrics.get("lufs")
        if isinstance(lv, (int, float)):
            lufs_vals.append(float(lv))
        tv = metrics.get("tempo_bpm")
        if isinstance(tv, (int, float)):
            tempo_vals.append(float(tv))
        kv = metrics.get("key")
        if isinstance(kv, str) and kv:
            key_vals.append(kv)
    mean_lufs = sum(lufs_vals) / len(lufs_vals) if lufs_vals else None
    mean_tempo = sum(tempo_vals) / len(tempo_vals) if tempo_vals else None
    key_summary = key_vals[0] if key_vals else None

    # Compute coarse lock summary at track level based on Song Graph targets.
    locks_meta: Dict[str, Any] = {}
    tempo_target = global_block.get("tempo_bpm")
    if isinstance(tempo_target, (int, float)) and mean_tempo is not None:
        tempo_target_f = float(tempo_target)
        if tempo_target_f > 0.0:
            tempo_score = 1.0 - (abs(float(mean_tempo) - tempo_target_f) / tempo_target_f)
            if tempo_score is not None:
                if tempo_score < 0.0:
                    tempo_score = 0.0
                if tempo_score > 1.0:
                    tempo_score = 1.0
                locks_meta["tempo_score"] = tempo_score
    key_target = global_block.get("key")
    if isinstance(key_target, str) and key_target.strip() and isinstance(key_summary, str) and key_summary:
        kt = key_target.strip().lower()
        ks = key_summary.strip().lower()
        locks_meta["key_score"] = 1.0 if kt == ks else 0.0
    if "tempo_score" in locks_meta and "key_score" in locks_meta:
        ts = locks_meta.get("tempo_score")
        ks = locks_meta.get("key_score")
        if isinstance(ts, (int, float)) and isinstance(ks, (int, float)):
            locks_meta["motif_lock"] = min(float(ts), float(ks))
    locks_meta.setdefault("lyrics_score", None)
    if locks_meta:
        components: List[float] = []
        for k in ("tempo_score", "key_score", "motif_lock", "lyrics_score"):
            v = locks_meta.get(k)
            if isinstance(v, (int, float)):
                components.append(float(v))
        if components:
            locks_meta["lock_score_music"] = min(components)

    # Song Graph summary for tracing (compact only).
    song_summary: Dict[str, Any] = {
        "sections_count": len(sections),
        "voices_count": len(music_branch.get("voices") or []),
        "instruments_count": len(music_branch.get("instruments") or []),
        "motifs_count": len(music_branch.get("motifs") or []),
    }
    # Track-level multi-axis evaluation (MusicEval 2.0) on the initial full track.
    track_eval = await _eval_full_track(full_path, music_branch, trace_id=trace_id, conversation_id=conversation_id)
    track_inner = track_eval.get("result") if isinstance(track_eval, dict) and isinstance(track_eval.get("result"), dict) else {}
    overall_block = track_inner.get("overall") if isinstance(track_inner.get("overall"), dict) else {}
    track_quality = overall_block.get("overall_quality_score")
    track_fit = overall_block.get("fit_score")

    # Identify windows that are weak according to per-window eval.
    bad_window_ids = _find_bad_windows(windows)

    # Regen loop: fix weak windows and optionally the whole track.
    regen_pass = 0
    while regen_pass < MAX_REGEN_PASSES:
        quality_ok = isinstance(track_quality, (int, float)) and float(track_quality) >= QUALITY_THRESHOLD_TRACK
        fit_ok = isinstance(track_fit, (int, float)) and float(track_fit) >= FIT_THRESHOLD_TRACK
        if quality_ok and fit_ok and not bad_window_ids:
            break
        log.warning(
            f"music.windowed.regen.pass conversation_id={conversation_id!r} trace_id={trace_id!r} pass={regen_pass!r} "
            f"track_quality={track_quality!r} track_fit={track_fit!r} bad_windows={bad_window_ids!r}"
        )
        if trace_id:
            try:
                trace_event(
                    "music.windowed.regen.pass",
                    {
                        "trace_id": trace_id,
                        "conversation_id": conversation_id,
                        "pass": regen_pass,
                        "track_quality": track_quality,
                        "track_fit": track_fit,
                        "bad_window_ids": bad_window_ids,
                    },
                )
            except Exception:
                log.debug(
                    f"music.windowed.trace_event_failed event={'music.windowed.regen.pass'!r} conversation_id={conversation_id!r} trace_id={trace_id!r}",
                    exc_info=True,
                )
        if bad_window_ids:
            windows, full_path = await _regenerate_bad_windows_only(
                provider,
                manifest,
                outdir_key,
                windows,
                bad_window_ids,
                music_branch,
                global_block,
                bpm,
                params,
                crossfade_frames,
                trace_id=trace_id,
                conversation_id=conversation_id,
                prompt=prompt,
                instrumental_only=instrumental_only,
                seed=seed,
            )
        else:
            windows, full_path = await _regenerate_full_music(
                provider,
                manifest,
                outdir_key,
                windows,
                music_branch,
                global_block,
                bpm,
                params,
                crossfade_frames,
                trace_id=trace_id,
                conversation_id=conversation_id,
                prompt=prompt,
                instrumental_only=instrumental_only,
                seed=seed,
            )
        track_eval = await _eval_full_track(full_path, music_branch, trace_id=trace_id, conversation_id=conversation_id)
        track_inner = track_eval.get("result") if isinstance(track_eval, dict) and isinstance(track_eval.get("result"), dict) else {}
        overall_block = track_inner.get("overall") if isinstance(track_inner.get("overall"), dict) else {}
        track_quality = overall_block.get("overall_quality_score")
        track_fit = overall_block.get("fit_score")
        bad_window_ids = _find_bad_windows(windows)
        regen_pass += 1

    # Style score against a style pack if attached to the music branch, computed on the final track.
    style_score = None
    sp = music_branch.get("style_pack") if isinstance(music_branch.get("style_pack"), dict) else None
    if sp is not None and isinstance(full_path, str) and full_path:
        style_score = style_score_for_track(full_path, sp)
        # Expose style_score on the locks meta block so downstream consumers and
        # dataset logging can reuse it without re-running CLAP.
        locks_meta["style_score"] = style_score

    append_training_sample(
        "music",
        {
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "tool": "music.infinite.windowed",
            "mode": mode,
            "prompt": prompt,
            "bpm": bpm,
            "length_s": length_s,
            "num_windows": len(windows),
            "lufs": mean_lufs,
            "tempo_bpm": mean_tempo,
            "key": key_summary,
            "lock_score_music": locks_meta.get("lock_score_music"),
            "song_graph_summary": song_summary,
            "music_profile_used": bool((global_block.get("style_tags") or [])),
            "style_score": style_score,
            "music_eval_track": track_eval,
            "bad_window_ids": bad_window_ids,
            "path": full_path,
        },
    )
    env: Dict[str, Any] = {
        "meta": {
            "model": (res.get("model") if isinstance(res, dict) else None) or "music",
            "ts": now_ts(),
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "step": 0,
            "state": "halt",
            "cont": {"present": False, "state_hash": None, "reason": None},
            "windows": windows,
            "lufs": mean_lufs,
            "tempo_bpm": mean_tempo,
            "key": key_summary,
            "locks": locks_meta,
        },
        "reasoning": {
            "goal": "music infinite windowed composition",
            "constraints": ["json-only", "edge-safe"],
            "decisions": ["music.infinite.windowed done"],
        },
        "evidence": [],
        "message": {"role": "assistant", "type": "tool", "content": "music windowed composed"},
        "tool_calls": [
            {
                "tool_name": "music.infinite.windowed",
                "tool": "music.infinite.windowed",
                "args": {
                    "prompt": prompt,
                    "bpm": bpm,
                    "length_s": length_s,
                    "window_bars": window_bars,
                    "overlap_bars": overlap_bars,
                },
                "status": "done",
                "artifact_id": artifact_id,
            }
        ],
        "artifacts": artifacts,
        "telemetry": {
            "window": {"input_bytes": 0, "output_target_tokens": 0},
            "compression_passes": [],
            "notes": [],
        },
    }
    # Normalize the already-constructed envelope dict without stringifying
    # and reparsing it, to avoid JSONParser noise and accidental shape drift.
    env = normalize_envelope(env)
    env = bump_envelope(env)
    assert_envelope(env)
    env = stamp_env(env, "music.infinite.windowed", env.get("meta", {}).get("model"))
    if trace_id:
        trace_event("tool.music.infinite.windowed.complete", {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "tool": "music.infinite.windowed",
            "path": full_path,
            "bytes": len(stitched_pcm),
            "windows": len(windows),
            "lufs": mean_lufs,
            "tempo_bpm": mean_tempo,
        })
    log.info(f"music.infinite.windowed: completed conversation_id={conversation_id!r} trace_id={trace_id!r} path={full_path!r} bytes={len(stitched_pcm)} windows={len(windows)}")
    # Expose only JSON-serializable audio metadata; raw PCM is persisted on disk
    # and referenced via artifacts/meta.
    env.setdefault("audio", {})
    if isinstance(env["audio"], dict):
        env["audio"].update(
            {
                "bytes": int(len(stitched_pcm)),
                "sample_rate": int(stitched_sr),
                "channels": int(stitched_ch),
            }
        )

    # Final dataset row (full fidelity tool result + artifacts)
    try:
        _ds_append_row(
            "tool_result",
            {
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "tool": "music.infinite.windowed",
                "tags": ["tool:music.infinite.windowed", "phase:finish"],
                "inputs": {
                    "prompt": prompt,
                    "mode": mode,
                    "length_s": length_s,
                    "bpm": bpm,
                    "window_bars": window_bars,
                    "overlap_bars": overlap_bars,
                    "seed": seed,
                },
                "outputs": {"path": full_path, "audio_ref": full_path},
                "locks": (lock_bundle if isinstance(lock_bundle, dict) else {}),
                "qa": track_eval if isinstance(track_eval, dict) else {},
                "metrics": {
                    "elapsed_ms": int((time.time() - t0_all) * 1000.0),
                    "track_quality": track_quality,
                    "track_fit": track_fit,
                    "bad_window_ids": bad_window_ids,
                    "windows": len(windows) if isinstance(windows, list) else 0,
                    "mean_lufs": mean_lufs,
                    "mean_tempo": mean_tempo,
                    "key": key_summary,
                    "style_score": style_score,
                },
                "meta": {"artifacts": artifacts},
                "payload": {"env": env},
            },
        )
    except Exception:
        log.debug(f"music.windowed.dataset.append_finish_failed conversation_id={conversation_id!r} trace_id={trace_id!r}", exc_info=True)

    # Final trace event
    if trace_id:
        try:
            trace_event(
                "music.windowed.finish",
                {
                    "trace_id": trace_id,
                    "conversation_id": conversation_id,
                    "full_path": full_path,
                    "elapsed_ms": int((time.time() - t0_all) * 1000.0),
                    "windows": len(windows) if isinstance(windows, list) else 0,
                    "track_quality": track_quality,
                    "track_fit": track_fit,
                    "bad_window_ids": bad_window_ids,
                },
            )
        except Exception:
            log.debug(
                f"music.windowed.trace_event_failed event={'music.windowed.finish'!r} conversation_id={conversation_id!r} trace_id={trace_id!r}",
                exc_info=True,
            )

    log.info(
        f"music.windowed.finish conversation_id={conversation_id!r} trace_id={trace_id!r} full_path={full_path!r} "
        f"elapsed_ms={int((time.time() - t0_all) * 1000.0)!r} windows={(len(windows) if isinstance(windows, list) else 0)!r} "
        f"track_quality={track_quality!r} track_fit={track_fit!r}"
    )
    return env


def restitch_music_from_windows(lock_bundle: Dict[str, Any], conversation_id: str, trace_id: str = "", crossfade_frames: int = 2048) -> Optional[str]:
    """
    Rebuild the full track from music.windows in the provided lock bundle.
    Returns the path to the stitched WAV on success, or None on failure.
    """
    music_branch = lock_bundle.get("music") if isinstance(lock_bundle.get("music"), dict) else {}
    windows = music_branch.get("windows") if isinstance(music_branch.get("windows"), list) else []
    if not windows:
        return None
    clips: List[Tuple[bytes, int, int]] = []
    # Use timing to sort windows for stitching.
    time_sorted_windows: List[Tuple[float, Dict[str, Any]]] = []
    for win in windows:
        if not isinstance(win, dict):
            continue
        t_start = win.get("t_start")
        try:
            t = float(t_start) if isinstance(t_start, (int, float)) else 0.0
        except Exception:
            t = 0.0
        time_sorted_windows.append((t, win))
    ranked_windows: List[Tuple[float, int, Dict[str, Any]]] = []
    for window_index, window_entry in enumerate(time_sorted_windows):
        ranked_windows.append((float(window_entry[0]), int(window_index), window_entry[1]))
    ranked_windows.sort()
    time_sorted_windows = [(ranked[0], ranked[2]) for ranked in ranked_windows]
    base_sr = None
    base_ch = None
    for _, win in time_sorted_windows:
        clip_path = win.get("artifact_path")
        if not (isinstance(clip_path, str) and clip_path and os.path.exists(clip_path)):
            continue
        try:
            with wave.open(clip_path, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
        except Exception:
            continue
        if base_sr is None:
            base_sr = sr
        if base_ch is None:
            base_ch = ch
        if sr != base_sr or ch != base_ch:
            continue
        clips.append((frames, sr, ch))
    if not clips:
        return None
    stitched_pcm, stitched_sr, stitched_ch = _stitch_windows_pcm(clips, crossfade_frames)
    outdir_key = conversation_id if isinstance(conversation_id, str) else ""
    outdir = os.path.join(UPLOAD_DIR, "artifacts", "music", outdir_key)
    ensure_dir(outdir)
    stem = f"windowed_restitch_{now_ts()}"
    full_path = os.path.join(outdir, f"{stem}.wav")
    with wave.open(full_path, "wb") as wf_full:
        wf_full.setnchannels(stitched_ch)
        wf_full.setsampwidth(2)
        wf_full.setframerate(stitched_sr)
        wf_full.writeframes(stitched_pcm)
    add_manifest_row({}, full_path, step_id="music.infinite.windowed.full")
    try:
        _ctx_add(outdir_key, "audio", full_path, None, None, ["music", "windowed"], {"trace_id": trace_id, "tool": "music.infinite.windowed"})
    except Exception as ex:
        log.warning(
            "music.windowed.ctx_add_error",
            extra={"conversation_id": outdir_key, "path": full_path},
            exc_info=ex,
        )
    return full_path


