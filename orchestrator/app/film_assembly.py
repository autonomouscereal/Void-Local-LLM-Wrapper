from __future__ import annotations

import base64
import json
import logging
import os
import glob
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx  # type: ignore

from .json_parser import JSONParser
from .trace_utils import emit_trace
from void_envelopes import _build_success_envelope, _build_error_envelope

log = logging.getLogger(__name__)


def _run_ffmpeg(cmd: List[str]) -> None:
    """
    Blocking ffmpeg runner. Raises on non-zero exit.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg_not_found")
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        log.error("ffmpeg.failed cmd=%r", cmd, exc_info=True)
        raise
    finally:
        log.info("ffmpeg.ok dur_ms=%d cmd0=%r", int((time.perf_counter() - t0) * 1000), cmd[:6])


def _write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _safe_public_url(public_base_url: str, abs_upload_path: str, *, upload_dir: str) -> str:
    if not isinstance(abs_upload_path, str) or not abs_upload_path:
        return ""
    if abs_upload_path.startswith("/workspace/uploads/"):
        rel = abs_upload_path.replace("/workspace", "")
        return f"{public_base_url.rstrip('/')}{rel}" if public_base_url else rel
    if abs_upload_path.startswith(upload_dir.rstrip("/") + "/"):
        rel2 = abs_upload_path.replace(upload_dir.rstrip("/"), "")
        if not rel2.startswith("/"):
            rel2 = "/" + rel2
        return f"{public_base_url.rstrip('/')}{rel2}" if public_base_url else rel2
    return abs_upload_path


def _parse_assets(assets: Any) -> Dict[str, Any]:
    if isinstance(assets, dict):
        return assets
    if isinstance(assets, str) and assets.strip():
        try:
            parser = JSONParser()
            parsed = parser.parse(assets, {})
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _mix_audio_wav(out_wav: str, *, tracks: List[str]) -> Optional[str]:
    tracks = [p for p in (tracks or []) if isinstance(p, str) and p and os.path.exists(p)]
    if not tracks:
        return None
    if len(tracks) == 1:
        return tracks[0]
    if not shutil.which("ffmpeg"):
        # Degrade safely: keep first track if ffmpeg is unavailable.
        log.warning("film_assembly.audio.mix.ffmpeg_missing using_first_track tracks=%d", len(tracks))
        return tracks[0]
    # Mix all tracks together, normalize later during mux if desired.
    cmd = ["ffmpeg", "-y"]
    for p in tracks:
        cmd += ["-i", p]
    # [0:a][1:a]... amix
    amix_in = "".join([f"[{i}:a]" for i in range(len(tracks))])
    cmd += [
        "-filter_complex",
        f"{amix_in}amix=inputs={len(tracks)}:duration=longest:dropout_transition=0[a]",
        "-map",
        "[a]",
        "-c:a",
        "pcm_s16le",
        out_wav,
    ]
    _run_ffmpeg(cmd)
    return out_wav


def _mux_frames_to_video(
    *,
    frames_glob: str,
    out_path: str,
    fps: int,
    audio_wav_path: Optional[str],
    subtitles_srt_path: Optional[str],
    upscale_scale: Optional[int],
    target_fps: Optional[int],
    audio_normalize: bool,
) -> None:
    # Validate frames exist up-front; ffmpeg glob otherwise fails late and opaquely.
    if not isinstance(frames_glob, str) or not frames_glob:
        raise ValueError("missing_frames_glob")
    if not glob.glob(frames_glob.replace("\\", "/")):
        raise FileNotFoundError(f"no_frames_for_glob:{frames_glob}")
    filters: List[str] = []
    if isinstance(upscale_scale, int) and upscale_scale in (2, 3, 4):
        filters.append(f"scale=iw*{upscale_scale}:ih*{upscale_scale}:flags=lanczos")
    if isinstance(target_fps, int) and target_fps > int(fps):
        filters.append(f"minterpolate=fps={int(target_fps)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    if subtitles_srt_path and os.path.exists(subtitles_srt_path):
        srt_posix = subtitles_srt_path.replace("\\", "/")
        filters.append(f"subtitles='{srt_posix}':force_style='FontName=Arial,FontSize=18'")
    vf = ["-vf", ",".join(filters)] if filters else []

    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-pattern_type",
        "glob",
        "-i",
        frames_glob,
    ]
    map_args: List[str] = ["-map", "0:v:0"]
    if audio_wav_path and os.path.exists(audio_wav_path):
        cmd += ["-i", audio_wav_path]
        map_args += ["-map", "1:a:0"]
    cmd += vf
    cmd += map_args
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
    ]
    if audio_wav_path and os.path.exists(audio_wav_path) and audio_normalize:
        cmd += ["-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += ["-shortest", out_path]
    _run_ffmpeg(cmd)


def _concat_videos_reencode(inputs: List[str], out_path: str) -> None:
    inputs = [p for p in (inputs or []) if isinstance(p, str) and p and os.path.exists(p)]
    if not inputs:
        raise ValueError("no_inputs")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpd:
        listfile = os.path.join(tmpd, "list.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for p in inputs:
                # ffmpeg concat demuxer file quoting: escape single quotes in the path.
                safe = p.replace("'", "'\\''")
                f.write("file '" + safe + "'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            listfile,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            out_path,
        ]
        _run_ffmpeg(cmd)


def _mux_audio_into_video(
    *,
    video_path: str,
    out_path: str,
    audio_path: str,
    policy: str,
) -> None:
    """
    policy:
      - 'replace': replace video audio with audio_path
      - 'mix': mix existing video audio with audio_path if present; else replace
      - 'keep': no-op (caller should just use video_path)
    """
    if not (isinstance(video_path, str) and video_path and os.path.exists(video_path)):
        raise FileNotFoundError(f"missing_video:{video_path}")
    if policy == "keep":
        if video_path != out_path:
            _run_ffmpeg(["ffmpeg", "-y", "-i", video_path, "-c", "copy", out_path])
        return
    if not (isinstance(audio_path, str) and audio_path and os.path.exists(audio_path)):
        if video_path != out_path:
            _run_ffmpeg(["ffmpeg", "-y", "-i", video_path, "-c", "copy", out_path])
        return
    if policy == "replace":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            out_path,
        ]
        _run_ffmpeg(cmd)
        return
    # mix
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        out_path,
    ]
    # If the input video has no audio stream, this will fail; caller can fall back to replace.
    try:
        _run_ffmpeg(cmd)
    except Exception:
        _mux_audio_into_video(video_path=video_path, out_path=out_path, audio_path=audio_path, policy="replace")


def _burn_subtitles(
    *,
    video_path: str,
    out_path: str,
    subtitles_srt_path: str,
) -> None:
    if not (isinstance(video_path, str) and video_path and os.path.exists(video_path)):
        raise FileNotFoundError(f"missing_video:{video_path}")
    if not (isinstance(subtitles_srt_path, str) and subtitles_srt_path and os.path.exists(subtitles_srt_path)):
        if video_path != out_path:
            _run_ffmpeg(["ffmpeg", "-y", "-i", video_path, "-c", "copy", out_path])
        return
    srt_posix = subtitles_srt_path.replace("\\", "/")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"subtitles='{srt_posix}':force_style='FontName=Arial,FontSize=18'",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        out_path,
    ]
    _run_ffmpeg(cmd)


def build_srt_from_dialogue_index(dialogue_index: Dict[str, Any]) -> str:
    """
    Build an SRT file from a dialogue_index mapping that contains entries with:
      - text
      - start_s
      - end_s
    """
    if not isinstance(dialogue_index, dict) or not dialogue_index:
        return ""
    # Collect valid entries, sort by start time
    rows: List[Tuple[float, float, str]] = []
    for _, v in dialogue_index.items():
        if not isinstance(v, dict):
            continue
        txt = v.get("text") or v.get("line") or v.get("content")
        if not isinstance(txt, str) or not txt.strip():
            continue
        s0 = v.get("start_s")
        s1 = v.get("end_s")
        if not isinstance(s0, (int, float)) or not isinstance(s1, (int, float)):
            continue
        if float(s1) <= float(s0):
            continue
        rows.append((float(s0), float(s1), txt.strip()))
    rows.sort(key=lambda r: (r[0], r[1]))

    def _fmt(t: float) -> str:
        if t < 0:
            t = 0.0
        h = int(t // 3600)
        t -= 3600 * h
        m = int(t // 60)
        t -= 60 * m
        s = int(t)
        ms = int((t - s) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    out_lines: List[str] = []
    idx = 1
    for s0, s1, txt in rows:
        out_lines.append(str(idx))
        out_lines.append(f"{_fmt(s0)} --> {_fmt(s1)}")
        out_lines.append(txt)
        out_lines.append("")
        idx += 1
    return "\n".join(out_lines)

async def assemble_film_from_scene_rows(
    *,
    film_id: str,
    scenes: List[Dict[str, Any]],
    upload_dir: str,
    public_base_url: str,
    preferences: Optional[Dict[str, Any]] = None,
    trace_id: str | None = None,
    state_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Canonical film assembly for the legacy scene-based pipeline.

    Each scene record expects:
      - assets: dict or JSON string containing:
          - urls: [{url, filename?}...] (image frames)
          - tts: {audio_wav_base64}? (optional)
          - music: {audio_wav_base64}? (optional)
          - subtitles_srt: str? (optional)

    preferences (all optional):
      - fps: int (default 24)
      - subtitles_enabled: bool (default False)
      - audio_enabled: bool (default True)
      - audio_normalize: bool (default True)
      - interpolation_enabled: bool + interpolation_target_fps: int
      - upscale_enabled: bool + upscale_scale: int (2/3/4)
    """
    prefs = preferences if isinstance(preferences, dict) else {}
    fps = int(prefs.get("fps") or 24)
    subtitles_enabled = bool(prefs.get("subtitles_enabled", False))
    audio_enabled = bool(prefs.get("audio_enabled", True))
    audio_normalize = bool(prefs.get("audio_normalize", True))
    interpolation_enabled = bool(prefs.get("interpolation_enabled", False))
    interpolation_target_fps = prefs.get("interpolation_target_fps")
    target_fps = int(interpolation_target_fps) if interpolation_enabled and isinstance(interpolation_target_fps, (int, float, str)) else None
    upscale_enabled = bool(prefs.get("upscale_enabled", False))
    upscale_scale_raw = prefs.get("upscale_scale")
    upscale_scale = int(upscale_scale_raw) if upscale_enabled and upscale_scale_raw in (2, 3, 4, "2", "3", "4") else None

    base_out = os.path.join(upload_dir, "artifacts", "video", "film_compile", str(film_id))
    os.makedirs(base_out, exist_ok=True)

    scene_outputs: List[Dict[str, Any]] = []
    scene_clip_paths: List[str] = []

    if isinstance(state_dir, str) and state_dir and isinstance(trace_id, str) and trace_id:
        try:
            emit_trace(state_dir, trace_id, "film.compile.start", {"film_id": film_id, "scenes_total": len(scenes or []), "preferences": prefs})
        except Exception:
            log.debug("film.compile.start trace failed (non-fatal)", exc_info=True)

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        for idx, sc in enumerate(scenes or []):
            if not isinstance(sc, dict):
                continue
            assets = _parse_assets(sc.get("assets"))
            urls = assets.get("urls") if isinstance(assets.get("urls"), list) else []
            frame_urls: List[str] = []
            for u in sorted([u for u in urls if isinstance(u, dict) and isinstance(u.get("url"), str) and u.get("url")], key=lambda x: str(x.get("filename") or "")):
                frame_urls.append(str(u.get("url")))
            if not frame_urls:
                continue

            with tempfile.TemporaryDirectory() as tmpd:
                # download frames
                frame_idx = 1
                for fu in frame_urls:
                    if not isinstance(fu, str) or not fu:
                        continue
                    try:
                        r = await client.get(fu)
                    except Exception:
                        log.warning("film.compile.frame_download_failed url=%r", fu, exc_info=True)
                        continue
                    if int(r.status_code) != 200:
                        continue
                    if not r.content:
                        continue
                    _write_bytes(os.path.join(tmpd, f"frame_{frame_idx:06d}.png"), bytes(r.content))
                    frame_idx += 1
                if frame_idx == 1:
                    continue

                # decode audio (optional)
                audio_wav: Optional[str] = None
                if audio_enabled:
                    tts = assets.get("tts") if isinstance(assets.get("tts"), dict) else {}
                    music = assets.get("music") if isinstance(assets.get("music"), dict) else {}
                    wav_paths: List[str] = []
                    for kind, blk in (("tts", tts), ("music", music)):
                        b64 = blk.get("audio_wav_base64")
                        if isinstance(b64, str) and b64.strip():
                            p = os.path.join(tmpd, f"{kind}.wav")
                            try:
                                _write_bytes(p, base64.b64decode(b64))
                            except Exception:
                                log.warning("film.compile.audio_decode_failed kind=%s", kind, exc_info=True)
                                continue
                            wav_paths.append(p)
                    mixed = _mix_audio_wav(os.path.join(tmpd, "mix.wav"), tracks=wav_paths)
                    audio_wav = mixed if isinstance(mixed, str) else None

                # subtitles (optional)
                srt_path: Optional[str] = None
                if subtitles_enabled:
                    srt = assets.get("subtitles_srt")
                    if isinstance(srt, str) and srt.strip():
                        srt_path = os.path.join(tmpd, "subtitles.srt")
                        _write_text(srt_path, srt)

                frames_glob = os.path.join(tmpd, "frame_*.png").replace("\\", "/")
                scene_out = os.path.join(base_out, f"scene_{idx:03d}.mp4")
                _mux_frames_to_video(
                    frames_glob=frames_glob,
                    out_path=scene_out,
                    fps=fps,
                    audio_wav_path=audio_wav,
                    subtitles_srt_path=srt_path,
                    upscale_scale=upscale_scale,
                    target_fps=target_fps,
                    audio_normalize=audio_normalize,
                )
                scene_clip_paths.append(scene_out)
                if isinstance(state_dir, str) and state_dir and isinstance(trace_id, str) and trace_id:
                    try:
                        emit_trace(
                            state_dir,
                            trace_id,
                            "film.compile.scene_done",
                            {
                                "film_id": film_id,
                                "scene_index": idx,
                                "scene_id": sc.get("id"),
                                "clip_path": scene_out,
                                "frames": frame_idx - 1,
                                "has_audio": bool(audio_wav),
                                "has_subtitles": bool(srt_path),
                            },
                        )
                    except Exception:
                        log.debug("film.compile.scene_done trace failed (non-fatal)", exc_info=True)
                scene_outputs.append(
                    {
                        "scene_index": idx,
                        "scene_id": sc.get("id"),
                        "index_num": sc.get("index_num"),
                        "clip_path": scene_out,
                        "clip_url": _safe_public_url(public_base_url, scene_out, upload_dir=upload_dir),
                        "has_audio": bool(audio_wav),
                        "has_subtitles": bool(srt_path),
                        "frames": frame_idx - 1,
                    }
                )

    if not scene_clip_paths:
        if isinstance(state_dir, str) and state_dir and isinstance(trace_id, str) and trace_id:
            try:
                emit_trace(state_dir, trace_id, "film.compile.failed", {"film_id": film_id, "reason": "no_scenes_with_frames"})
            except Exception:
                log.debug("film.compile.failed trace failed (non-fatal)", exc_info=True)
        return _build_error_envelope(
            "no_scenes_with_frames",
            "No scenes contained downloadable frames in assets.urls",
            f"film.compile:{film_id}",
            status=422,
            details={"film_id": film_id},
        )

    # Canonical output under artifacts; best-effort convenience copy under uploads root.
    final_out_art = os.path.join(base_out, "film.mp4")
    _concat_videos_reencode(scene_clip_paths, final_out_art)
    final_out_public = os.path.join(upload_dir, f"film_{film_id}.mp4")
    try:
        _run_ffmpeg(["ffmpeg", "-y", "-i", final_out_art, "-c", "copy", final_out_public])
    except Exception:
        final_out_public = final_out_art

    if isinstance(state_dir, str) and state_dir and isinstance(trace_id, str) and trace_id:
        try:
            emit_trace(state_dir, trace_id, "film.compile.done", {"film_id": film_id, "path": final_out_art, "public_path": final_out_public, "scenes": len(scene_clip_paths)})
        except Exception:
            log.debug("film.compile.done trace failed (non-fatal)", exc_info=True)
    rid = f"film.compile:{film_id}"
    return _build_success_envelope(
        {
            "film_id": film_id,
            "url": _safe_public_url(public_base_url, final_out_public, upload_dir=upload_dir),
            "path": final_out_art,
            "public_path": final_out_public,
            "scenes": scene_outputs,
            "preferences": {
                "fps": fps,
                "subtitles_enabled": subtitles_enabled,
                "audio_enabled": audio_enabled,
                "interpolation_target_fps": target_fps,
                "upscale_scale": upscale_scale,
            },
        },
        rid,
    )


def assemble_film_from_shots(
    *,
    film_id: str,
    shots: List[Dict[str, Any]],
    upload_dir: str,
    public_base_url: str,
    audio_path: Optional[str] = None,
    audio_policy: str = "replace",
    subtitles_srt_path: Optional[str] = None,
    trace_id: str | None = None,
    state_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Canonical assembly for Film2: take ordered shot video paths and produce:
      - per-scene clips (grouped by scene_id when available)
      - final film mp4

    audio_policy: replace|mix|keep
    """
    base_out = os.path.join(upload_dir, "artifacts", "video", "film2_assemble", str(film_id))
    os.makedirs(base_out, exist_ok=True)
    if isinstance(state_dir, str) and state_dir and isinstance(trace_id, str) and trace_id:
        try:
            emit_trace(state_dir, trace_id, "film2.assemble.start", {"film_id": film_id, "shots_total": len(shots or []), "audio_policy": audio_policy, "audio_path": audio_path, "subtitles": bool(subtitles_srt_path)})
        except Exception:
            log.debug("film2.assemble.start trace failed (non-fatal)", exc_info=True)

    # Build ordered scene groups
    scene_order: List[str] = []
    by_scene: Dict[str, List[str]] = {}
    for sh in (shots or []):
        if not isinstance(sh, dict):
            continue
        sid = sh.get("scene_id")
        scene_id = sid if isinstance(sid, str) and sid else "scene_0"
        # choose best available per-shot path
        p = (
            sh.get("best_path")
            or sh.get("final_clean_path")
            or sh.get("final_path")
            or sh.get("upscaled_path")
            or sh.get("interp_path")
            or sh.get("clean_path")
            or sh.get("gen_path")
        )
        if not (isinstance(p, str) and p and os.path.exists(p)):
            continue
        if scene_id not in by_scene:
            by_scene[scene_id] = []
            scene_order.append(scene_id)
        by_scene[scene_id].append(p)

    scene_clips: List[Dict[str, Any]] = []
    scene_clip_paths: List[str] = []
    for idx, scene_id in enumerate(scene_order):
        parts = by_scene.get(scene_id) or []
        if not parts:
            continue
        out_scene = os.path.join(base_out, f"{scene_id}_{idx:03d}.mp4")
        _concat_videos_reencode(parts, out_scene)
        scene_clip_paths.append(out_scene)
        if isinstance(state_dir, str) and state_dir and isinstance(trace_id, str) and trace_id:
            try:
                emit_trace(state_dir, trace_id, "film2.assemble.scene_done", {"film_id": film_id, "scene_id": scene_id, "scene_index": idx, "clip_path": out_scene, "shot_count": len(parts)})
            except Exception:
                log.debug("film2.assemble.scene_done trace failed (non-fatal)", exc_info=True)
        scene_clips.append(
            {
                "scene_id": scene_id,
                "scene_index": idx,
                "clip_path": out_scene,
                "clip_url": _safe_public_url(public_base_url, out_scene, upload_dir=upload_dir),
                "shot_count": len(parts),
            }
        )

    if not scene_clip_paths:
        if isinstance(state_dir, str) and state_dir and isinstance(trace_id, str) and trace_id:
            try:
                emit_trace(state_dir, trace_id, "film2.assemble.failed", {"film_id": film_id, "reason": "no_shot_videos"})
            except Exception:
                log.debug("film2.assemble.failed trace failed (non-fatal)", exc_info=True)
        return _build_error_envelope(
            "no_shot_videos",
            "No shot videos were available to assemble",
            f"film2.assemble:{film_id}",
            status=422,
            details={"film_id": film_id},
        )

    stitched = os.path.join(base_out, "film_noaudio.mp4")
    _concat_videos_reencode(scene_clip_paths, stitched)

    final_out = os.path.join(upload_dir, "artifacts", "video", "film2_assemble", str(film_id), "film.mp4")
    os.makedirs(os.path.dirname(final_out), exist_ok=True)
    tmp_mux = os.path.join(base_out, "film_mux.mp4")
    _mux_audio_into_video(video_path=stitched, out_path=tmp_mux, audio_path=(audio_path or ""), policy=str(audio_policy or "replace"))
    if subtitles_srt_path and isinstance(subtitles_srt_path, str) and subtitles_srt_path:
        _burn_subtitles(video_path=tmp_mux, out_path=final_out, subtitles_srt_path=subtitles_srt_path)
    else:
        if tmp_mux != final_out:
            _run_ffmpeg(["ffmpeg", "-y", "-i", tmp_mux, "-c", "copy", final_out])

    if isinstance(state_dir, str) and state_dir and isinstance(trace_id, str) and trace_id:
        try:
            emit_trace(state_dir, trace_id, "film2.assemble.done", {"film_id": film_id, "path": final_out, "scenes": len(scene_clip_paths)})
        except Exception:
            log.debug("film2.assemble.done trace failed (non-fatal)", exc_info=True)
    rid2 = f"film2.assemble:{film_id}"
    return _build_success_envelope(
        {
            "film_id": film_id,
            "path": final_out,
            "url": _safe_public_url(public_base_url, final_out, upload_dir=upload_dir),
            "scenes": scene_clips,
            "audio_policy": audio_policy,
            "audio_path": audio_path,
        },
        rid2,
    )


