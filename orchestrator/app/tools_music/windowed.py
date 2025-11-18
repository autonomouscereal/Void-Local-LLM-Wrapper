from __future__ import annotations

import os
import wave
import logging
from typing import Any, Dict, List, Tuple, Optional

from .common import now_ts, ensure_dir, sidecar, stamp_env, music_edge_defaults
from ..analysis.media import analyze_audio
from ..music.style_pack import style_score_for_track  # type: ignore
from ..music.eval import compute_music_eval  # type: ignore
from ..artifacts.manifest import add_manifest_row
from ..context.index import add_artifact as _ctx_add
from ..datasets.trace import append_sample as _trace_append
from ..jsonio.normalize import normalize_to_envelope
from ..jsonio.versioning import bump_envelope, assert_envelope


# Regeneration thresholds and limits for MusicEval-driven windowed composition.
QUALITY_THRESHOLD_TRACK = 0.7
FIT_THRESHOLD_TRACK = 0.7
WINDOW_QUALITY_THRESHOLD = 0.6
WINDOW_FIT_THRESHOLD = 0.6
MAX_REGEN_PASSES = 2

# Hard cap for per-window generation length passed to the backend music engine.
# The infinite/windowed tool is responsible for stitching longer tracks.
MAX_WINDOW_SECONDS = 20


def _eval_window_clip(
    win: Dict[str, Any],
    clip_path: str,
    cid: str,
    music_branch: Dict[str, Any],
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
    eval_out = compute_music_eval(
        track_path=clip_path,
        song_graph=local_song_graph,
        style_pack=style_pack,
        film_context=film_context,
    )
    overall = eval_out.get("overall") if isinstance(eval_out.get("overall"), dict) else {}
    win["eval"] = eval_out
    win["quality_score"] = overall.get("overall_quality_score")
    win["fit_score"] = overall.get("fit_score")
    _trace_append(
        "music",
        {
            "event": "music.window.eval",
            "cid": cid,
            "tool": "music.infinite.windowed",
            "window_id": win.get("window_id"),
            "section_id": win.get("section_id"),
            "eval": eval_out,
            "path": clip_path,
        },
    )
    return eval_out


def _eval_full_track(full_path: str, music_branch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the stitched full track for multi-axis MusicEval.
    """
    style_pack = music_branch.get("style_pack") if isinstance(music_branch.get("style_pack"), dict) else None
    track_eval = compute_music_eval(
        track_path=full_path,
        song_graph=music_branch,
        style_pack=style_pack,
        film_context={},
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


def _recompose_single_window(
    job: Dict[str, Any],
    provider,
    manifest: Dict[str, Any],
    cid: str,
    win: Dict[str, Any],
    music_branch: Dict[str, Any],
    global_block: Dict[str, Any],
    bpm: int,
    params: Dict[str, Any],
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
    prompt = job.get("prompt") or ""
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
    if bool(job.get("instrumental_only")):
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
        "seed": job.get("seed"),
    }
    res = provider.compose(compose_args)
    wav_bytes = res.get("wav_bytes") or b""
    clip_stem = f"{window_id}_{now_ts()}"
    outdir = os.path.join("/workspace", "uploads", "artifacts", "music", cid)
    ensure_dir(outdir)
    clip_path = os.path.join(outdir, f"{clip_stem}.wav")
    with wave.open(clip_path, "wb") as wf:
        wf.setnchannels(ch_local)
        wf.setsampwidth(2)
        wf.setframerate(sr_local)
        wf.writeframes(wav_bytes)
    add_manifest_row(manifest, clip_path, step_id="music.infinite.windowed")
    ainfo = analyze_audio(clip_path)
    tempo_detected = ainfo.get("tempo_bpm")
    tempo_target = None
    if isinstance(sec_obj, dict) and isinstance(sec_obj.get("tempo_bpm"), (int, float)):
        tempo_target = float(sec_obj.get("tempo_bpm"))
    elif isinstance(global_block.get("tempo_bpm"), (int, float)):
        tempo_target = float(global_block.get("tempo_bpm"))
    tempo_lock = None
    if isinstance(tempo_detected, (int, float)) and tempo_target and tempo_target > 0.0:
        tempo_lock = 1.0 - (abs(float(tempo_detected) - tempo_target) / tempo_target)
        if tempo_lock < 0.0:
            tempo_lock = 0.0
        if tempo_lock > 1.0:
            tempo_lock = 1.0
    key_detected = ainfo.get("key")
    key_target = None
    if isinstance(sec_obj, dict) and isinstance(sec_obj.get("key"), str):
        key_target = sec_obj.get("key")
    elif isinstance(global_block.get("key"), str):
        key_target = global_block.get("key")
    key_lock = None
    if (
        isinstance(key_target, str)
        and key_target.strip()
        and isinstance(key_detected, str)
        and key_detected
    ):
        key_lock = 1.0 if key_target.strip().lower() == key_detected.strip().lower() else 0.0
    lyrics_lock = None
    motif_lock = None
    if isinstance(tempo_lock, (int, float)) and isinstance(key_lock, (int, float)):
        motif_lock = min(float(tempo_lock), float(key_lock))
    ainfo["tempo_lock"] = tempo_lock
    ainfo["key_lock"] = key_lock
    ainfo["lyrics_lock"] = lyrics_lock
    ainfo["motif_lock"] = motif_lock
    win["artifact_path"] = clip_path
    win["metrics"] = ainfo
    _eval_window_clip(win, clip_path, cid, music_branch)
    return wav_bytes, sr_local, ch_local


def _stitch_from_windows(
    windows: List[Dict[str, Any]],
    crossfade_frames: int,
) -> Tuple[bytes, int, int]:
    """
    Helper to rebuild PCM from existing windows on disk using timing order.
    """
    clips: List[Tuple[bytes, int, int]] = []
    sortable: List[Tuple[float, Dict[str, Any]]] = []
    for win in windows:
        if not isinstance(win, dict):
            continue
        t_start = win.get("t_start")
        try:
            t_val = float(t_start) if isinstance(t_start, (int, float)) else 0.0
        except Exception:
            t_val = 0.0
        sortable.append((t_val, win))
    sortable.sort(key=lambda x: x[0])
    base_sr: Optional[int] = None
    base_ch: Optional[int] = None
    for _, win in sortable:
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
        return b"", 44100, 2
    stitched_pcm, stitched_sr, stitched_ch = _stitch_windows_pcm(clips, crossfade_frames)
    return stitched_pcm, stitched_sr, stitched_ch


def _regenerate_bad_windows_only(
    job: Dict[str, Any],
    provider,
    manifest: Dict[str, Any],
    cid: str,
    windows: List[Dict[str, Any]],
    bad_window_ids: List[str],
    music_branch: Dict[str, Any],
    global_block: Dict[str, Any],
    bpm: int,
    params: Dict[str, Any],
    crossfade_frames: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Recompose only windows listed in bad_window_ids, re-evaluate them, and stitch a new full track.
    """
    bad_set = {wid for wid in bad_window_ids if isinstance(wid, str) and wid}
    for win in windows:
        wid = win.get("window_id")
        if isinstance(wid, str) and wid in bad_set:
            _recompose_single_window(job, provider, manifest, cid, win, music_branch, global_block, bpm, params)
    stitched_pcm, stitched_sr, stitched_ch = _stitch_from_windows(windows, crossfade_frames)
    outdir = os.path.join("/workspace", "uploads", "artifacts", "music", cid)
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


def _regenerate_full_music(
    job: Dict[str, Any],
    provider,
    manifest: Dict[str, Any],
    cid: str,
    windows: List[Dict[str, Any]],
    music_branch: Dict[str, Any],
    global_block: Dict[str, Any],
    bpm: int,
    params: Dict[str, Any],
    crossfade_frames: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Rebuild all windows with provider.compose, re-evaluate, and stitch into a fresh full track.
    """
    new_clips: List[Tuple[bytes, int, int, Dict[str, Any]]] = []
    for win in windows:
        if not isinstance(win, dict):
            continue
        wav_bytes, sr_local, ch_local = _recompose_single_window(
            job,
            provider,
            manifest,
            cid,
            win,
            music_branch,
            global_block,
            bpm,
            params,
        )
        new_clips.append((wav_bytes, sr_local, ch_local, win))
    clip_pcm_list: List[Tuple[bytes, int, int]] = []
    for pcm, sr, ch, _ in new_clips:
        clip_pcm_list.append((pcm, sr, ch))
    stitched_pcm, stitched_sr, stitched_ch = _stitch_windows_pcm(clip_pcm_list, crossfade_frames)
    outdir = os.path.join("/workspace", "uploads", "artifacts", "music", cid)
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
    bpm_val = int(bpm) if bpm and bpm > 0 else 120
    bars_total = max(8, int(round((seconds / 60.0) * bpm_val / 4.0)))
    return {
        "music_id": "main_theme",
        "reference_tracks": [],
        "text_prompt": prompt,
        "genre_tags": [],
        "mood_tags": [],
        "embeddings": {},
        "tempo_bpm": bpm_val,
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
        for i in range(0, fade_bytes, 2):
            pos_prev = start_prev + i
            pos_new = i
            s_prev = int.from_bytes(out[pos_prev:pos_prev + 2], "little", signed=True)
            s_new = int.from_bytes(pcm[pos_new:pos_new + 2], "little", signed=True)
            frac = i / max(2, float(fade_bytes))
            mix = int(s_prev * (1.0 - frac) + s_new * frac)
            if mix < -32768:
                mix = -32768
            if mix > 32767:
                mix = 32767
            blended[pos_new:pos_new + 2] = int(mix).to_bytes(2, "little", signed=True)
        out[start_prev:] = blended
        out.extend(pcm[fade_bytes:])
    return bytes(out), base_sr, base_ch


def run_music_infinite_windowed(job: Dict[str, Any], provider, manifest: Dict[str, Any]) -> Dict[str, Any]:
    cid = job.get("cid") or f"music-{now_ts()}"
    prompt = job.get("prompt") or ""
    outdir = os.path.join("/workspace", "uploads", "artifacts", "music", cid)
    ensure_dir(outdir)
    params = music_edge_defaults(
        {
            "bpm": job.get("bpm"),
            "length_s": job.get("length_s"),
            "sample_rate": job.get("sample_rate"),
            "channels": job.get("channels"),
        },
        edge=bool(job.get("edge")),
    )
    mode = str(job.get("mode") or "start").strip().lower()
    length_s = int(params.get("length_s") or 60)
    bpm = int(params.get("bpm") or 120)
    window_bars = int(job.get("window_bars") or 8)
    overlap_bars = int(job.get("overlap_bars") or 1)
    lock_bundle = job.get("lock_bundle") if isinstance(job.get("lock_bundle"), dict) else {}
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
    instrumental_only = bool(job.get("instrumental_only"))
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
                window_clips.append((b"", sr_local, ch_local, win))
                continue
        existing_path = win.get("artifact_path")
        if isinstance(existing_path, str) and existing_path:
            # Existing window from a prior run; rely on disk contents and metrics.
            window_clips.append((b"", sr_local, ch_local, win))
            continue
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
            "seed": job.get("seed"),
        }
        res = provider.compose(compose_args)
        wav_bytes = res.get("wav_bytes") or b""
        clip_stem = f"{window_id}_{now_ts()}"
        clip_path = os.path.join(outdir, f"{clip_stem}.wav")
        with wave.open(clip_path, "wb") as wf:
            wf.setnchannels(ch_local)
            wf.setsampwidth(2)
            wf.setframerate(sr_local)
            wf.writeframes(wav_bytes)
        add_manifest_row(manifest, clip_path, step_id="music.infinite.windowed")
        ainfo = analyze_audio(clip_path)
        # Per-window lock scores based on Song Graph targets.
        # Tempo lock: closeness of detected tempo to section/global tempo.
        tempo_detected = ainfo.get("tempo_bpm")
        sec_obj = section_index.get(win.get("section_id")) if isinstance(win.get("section_id"), str) else None
        tempo_target = None
        if isinstance(sec_obj, dict) and isinstance(sec_obj.get("tempo_bpm"), (int, float)):
            tempo_target = float(sec_obj.get("tempo_bpm"))
        elif isinstance(global_block.get("tempo_bpm"), (int, float)):
            tempo_target = float(global_block.get("tempo_bpm"))
        tempo_lock = None
        if isinstance(tempo_detected, (int, float)) and tempo_target and tempo_target > 0.0:
            try:
                tempo_lock = 1.0 - (abs(float(tempo_detected) - tempo_target) / tempo_target)
                if tempo_lock < 0.0:
                    tempo_lock = 0.0
                if tempo_lock > 1.0:
                    tempo_lock = 1.0
            except Exception:
                tempo_lock = None
        key_detected = ainfo.get("key")
        key_target = None
        if isinstance(sec_obj, dict) and isinstance(sec_obj.get("key"), str):
            key_target = sec_obj.get("key")
        elif isinstance(global_block.get("key"), str):
            key_target = global_block.get("key")
        key_lock = None
        if isinstance(key_target, str) and key_target.strip() and isinstance(key_detected, str) and key_detected:
            key_lock = 1.0 if key_target.strip().lower() == key_detected.strip().lower() else 0.0
        # Lyrics/motif locks are placeholders until detailed alignment/motif detection is wired.
        lyrics_lock = None
        motif_lock = None
        if isinstance(tempo_lock, (int, float)) and isinstance(key_lock, (int, float)):
            motif_lock = min(float(tempo_lock), float(key_lock))
        ainfo["tempo_lock"] = tempo_lock
        ainfo["key_lock"] = key_lock
        ainfo["lyrics_lock"] = lyrics_lock
        ainfo["motif_lock"] = motif_lock
        win["artifact_path"] = clip_path
        win["metrics"] = ainfo
        # Per-window multi-axis evaluation (MusicEval 2.0).
        _eval_window_clip(win, clip_path, cid, music_branch)
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
    crossfade_frames = int(job.get("crossfade_frames") or 2048)
    stitched_pcm, stitched_sr, stitched_ch = _stitch_windows_pcm(
        clip_pcm_list,
        crossfade_frames,
    )
    stem = f"windowed_{now_ts()}"
    full_path = os.path.join(outdir, f"{stem}.wav")
    with wave.open(full_path, "wb") as wf_full:
        wf_full.setnchannels(stitched_ch)
        wf_full.setsampwidth(2)
        wf_full.setframerate(stitched_sr)
        wf_full.writeframes(stitched_pcm)
    add_manifest_row(manifest, full_path, step_id="music.infinite.windowed.full")
    _ctx_add(cid, "audio", full_path, None, None, ["music", "windowed"], {})
    artifacts: List[Dict[str, Any]] = [
        {"id": os.path.basename(full_path), "kind": "audio-ref", "summary": stem, "bytes": len(stitched_pcm)}
    ]
    for _, _, _, win in window_clips:
        clip_path = win.get("artifact_path")
        if isinstance(clip_path, str):
            artifacts.append(
                {
                    "id": os.path.basename(clip_path),
                    "kind": "audio-window",
                    "summary": win.get("window_id"),
                    "bytes": os.path.getsize(clip_path) if os.path.exists(clip_path) else 0,
                }
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
    track_eval = _eval_full_track(full_path, music_branch)
    overall_block = track_eval.get("overall") if isinstance(track_eval.get("overall"), dict) else {}
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
        if bad_window_ids:
            windows, full_path = _regenerate_bad_windows_only(
                job,
                provider,
                manifest,
                cid,
                windows,
                bad_window_ids,
                music_branch,
                global_block,
                bpm,
                params,
                crossfade_frames,
            )
        else:
            windows, full_path = _regenerate_full_music(
                job,
                provider,
                manifest,
                cid,
                windows,
                music_branch,
                global_block,
                bpm,
                params,
                crossfade_frames,
            )
        track_eval = _eval_full_track(full_path, music_branch)
        overall_block = track_eval.get("overall") if isinstance(track_eval.get("overall"), dict) else {}
        track_quality = overall_block.get("overall_quality_score")
        track_fit = overall_block.get("fit_score")
        bad_window_ids = _find_bad_windows(windows)
        regen_pass += 1

    # Style score against a style pack if attached to the music branch, computed on the final track.
    style_score = None
    sp = music_branch.get("style_pack") if isinstance(music_branch.get("style_pack"), dict) else None
    if sp is not None and isinstance(full_path, str) and full_path:
        style_score = style_score_for_track(full_path, sp)

    _trace_append(
        "music",
        {
            "event": "music.infinite.windowed",
            "cid": cid,
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
            "model": res.get("model", "music"),
            "ts": now_ts(),
            "cid": cid,
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
                "tool": "music.infinite.windowed",
                "args": {
                    "prompt": prompt,
                    "bpm": bpm,
                    "length_s": length_s,
                    "window_bars": window_bars,
                    "overlap_bars": overlap_bars,
                },
                "status": "done",
                "result_ref": os.path.basename(full_path),
            }
        ],
        "artifacts": artifacts,
        "telemetry": {
            "window": {"input_bytes": 0, "output_target_tokens": 0},
            "compression_passes": [],
            "notes": [],
        },
    }
    env = normalize_to_envelope(env)
    env = bump_envelope(env)
    assert_envelope(env)
    env = stamp_env(env, "music.infinite.windowed", env.get("meta", {}).get("model"))
    env["wav_bytes"] = stitched_pcm
    return env


def restitch_music_from_windows(lock_bundle: Dict[str, Any], cid: str, crossfade_frames: int = 2048) -> Optional[str]:
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
    sortable: List[Tuple[float, Dict[str, Any]]] = []
    for win in windows:
        if not isinstance(win, dict):
            continue
        t_start = win.get("t_start")
        try:
            t_val = float(t_start) if isinstance(t_start, (int, float)) else 0.0
        except Exception:
            t_val = 0.0
        sortable.append((t_val, win))
    sortable.sort(key=lambda x: x[0])
    base_sr = None
    base_ch = None
    for _, win in sortable:
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
    outdir = os.path.join("/workspace", "uploads", "artifacts", "music", cid)
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
        _ctx_add(cid, "audio", full_path, None, None, ["music", "windowed"], {})
    except Exception:
        pass
    return full_path


