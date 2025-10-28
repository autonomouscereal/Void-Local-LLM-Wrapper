from __future__ import annotations
# HARD BAN (permanent): No Pydantic, no SQLAlchemy/ORM, no CSV/Parquet. JSON/NDJSON only.

import os
import json
import subprocess
import tempfile
from typing import Dict, Any, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import base64
import requests
from urllib.parse import urlparse, parse_qs
from .json_parser import JSONParser


PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
WORKSPACE = "/workspace"

app = FastAPI(title="Film Assembler", version="0.1.0")


def _safe_path(name: str) -> str:
    return os.path.join(WORKSPACE, "uploads", name)


def _ffmpeg_mux(
    image_glob: str,
    audio_wav_path: str | None,
    srt_path: str | None,
    out_path: str,
    fps: int = 24,
    target_fps: int | None = None,
    upscale_scale: int | None = None,
    audio_normalize: bool = True,
) -> None:
    filters: List[str] = []
    inputs: List[str] = []
    # images
    inputs += ["-framerate", str(fps), "-pattern_type", "glob", "-i", image_glob]
    map_args: List[str] = ["-map", "0:v:0"]
    if audio_wav_path:
        inputs += ["-i", audio_wav_path]
        map_args += ["-map", "1:a:0"]
    # upscale via scale filter if requested
    if upscale_scale and upscale_scale in (2, 3, 4):
        filters.append(f"scale=iw*{upscale_scale}:ih*{upscale_scale}:flags=lanczos")
    # interpolate frames using ffmpeg's motion compensation if target_fps > fps
    if target_fps and target_fps > fps:
        filters.append(f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    if srt_path and os.path.exists(srt_path):
        # burn-in subtitles (avoid backslashes directly in f-string expressions)
        srt_posix = srt_path.replace('\\', '/')
        filters.append(f"subtitles='{srt_posix}':force_style='FontName=Arial,FontSize=18'")
    vf = []
    if filters:
        vf = ["-vf", ",".join(filters)]
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        *vf,
        *map_args,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        *( ["-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11"] if audio_wav_path and audio_normalize else [] ),
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def _download(url: str, dest_path: str, timeout: int = 60) -> None:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)


@app.post("/assemble")
async def assemble(body: Dict[str, Any]):
    film_id = body.get("film_id")
    scenes = body.get("scenes") or []
    prefs = body.get("preferences") or {}
    fps = int(prefs.get("fps") or 24)
    if not film_id or not scenes:
        return JSONResponse(status_code=400, content={"error": "missing film_id or scenes"})
    # Build a temp workspace for frames/audio/subtitles
    with tempfile.TemporaryDirectory() as tmpd:
        # 1) Collect frame URLs from scenes.assets.urls (ComfyUI /view endpoints)
        frame_idx = 1
        for sc in scenes:
            assets = sc.get("assets") or {}
            if isinstance(assets, str):
                # Use hardened JSON parser with expected structure
                try:
                    parser = JSONParser()
                    expected = {"urls": list, "outputs": dict, "status": dict}
                    assets = parser.parse(assets, expected)
                except Exception:
                    assets = {}
            urls = assets.get("urls") or []
            # Try to sort predictably by filename within each scene
            def _key(u: Dict[str, Any]):
                fn = u.get("filename") or ""
                return fn
            for u in sorted([u for u in urls if isinstance(u, dict) and u.get("url")], key=_key):
                img_url = u.get("url")
                if not isinstance(img_url, str):
                    continue
                # Only include images; ComfyUI view returns images for image outputs
                fname = os.path.join(tmpd, f"frame_{frame_idx:06d}.png")
                try:
                    _download(img_url, fname)
                    frame_idx += 1
                except Exception:
                    continue

        if frame_idx == 1:
            return JSONResponse(status_code=400, content={"error": "no frames found in scenes.assets.urls"})

        # 2) Audio: prefer TTS over music; decode base64 if present
        audio_path = None
        for sc in scenes:
            assets = sc.get("assets") or {}
            if isinstance(assets, str):
                try:
                    parser = JSONParser()
                    expected = {"tts": dict, "music": dict, "urls": list}
                    assets = parser.parse(assets, expected)
                except Exception:
                    assets = {}
            tts = assets.get("tts") or {}
            music = assets.get("music") or {}
            data_b64 = tts.get("audio_wav_base64") or music.get("audio_wav_base64")
            if isinstance(data_b64, str) and data_b64.strip():
                audio_path = os.path.join(tmpd, "audio.wav")
                try:
                    with open(audio_path, "wb") as f:
                        f.write(base64.b64decode(data_b64))
                except Exception:
                    audio_path = None
                break

        # 3) Subtitles: simple SRT string if present
        srt_path = None
        for sc in scenes:
            assets = sc.get("assets") or {}
            if isinstance(assets, str):
                try:
                    parser = JSONParser()
                    expected = {"subtitles_srt": str, "urls": list}
                    assets = parser.parse(assets, expected)
                except Exception:
                    assets = {}
            srt = assets.get("subtitles_srt")
            if isinstance(srt, str) and srt.strip():
                srt_path = os.path.join(tmpd, "subtitles.srt")
                try:
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(srt)
                except Exception:
                    srt_path = None
                break

        # 4) Run ffmpeg
        image_glob = os.path.join(tmpd, "frame_*.png")
        out_name = f"film_{film_id}.mp4"
        out_path = _safe_path(out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            # Optional post-processing preferences
            target_fps = None
            upscale_scale = None
            if isinstance(prefs, dict):
                if prefs.get("interpolation_enabled") and prefs.get("interpolation_target_fps"):
                    try:
                        target_fps = int(prefs.get("interpolation_target_fps"))
                    except Exception:
                        target_fps = None
                if prefs.get("upscale_enabled") and prefs.get("upscale_scale") in (2, 3, 4):
                    upscale_scale = int(prefs.get("upscale_scale"))
            _ffmpeg_mux(
                image_glob=image_glob,
                audio_wav_path=audio_path,
                srt_path=srt_path,
                out_path=out_path,
                fps=fps,
                target_fps=target_fps,
                upscale_scale=upscale_scale,
            )
            url = f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{out_name}" if PUBLIC_BASE_URL else f"/uploads/{out_name}"
            return {"url": url}
        except subprocess.CalledProcessError as ex:
            return JSONResponse(status_code=500, content={"error": str(ex)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


