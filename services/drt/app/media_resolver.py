from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from void_json.json_parser import JSONParser


COOKIES_FILE = os.getenv("MEDIA_COOKIES_FILE", "")
HEADERS_JSON = os.getenv("MEDIA_HEADERS_JSON", "")
DEFAULT_USER_AGENT = "Void-Wrapper/1.0 (+local)"

YOUTUBE_RX = re.compile(r"(youtube\.com|youtu\.be)/", re.I)
GENERIC_RX = re.compile(r"\.(mp4|m4a|webm|mp3|wav|aac|m3u8)(\?|$)", re.I)
PDF_RX = re.compile(r"\.pdf(\?|$)", re.I)
IMAGE_RX = re.compile(r"\.(png|jpe?g|gif|webp|bmp|tiff?)(\?|$)", re.I)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _yt_dlp_info(url: str) -> Dict[str, Any]:
    cmd = [
        "yt-dlp",
        "-J",
        "--no-check-certificate",
        "--no-warnings",
        "--ignore-config",
        "--no-playlist",
        url,
    ]
    if COOKIES_FILE:
        cmd += ["--cookies", COOKIES_FILE]
    if HEADERS_JSON:
        parser = JSONParser()
        hdrs = parser.parse(HEADERS_JSON, {}) or {}
        if isinstance(hdrs, dict):
            for k, v in hdrs.items():
                cmd += ["--add-header", f"{k}:{v}"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    parser = JSONParser()
    obj = parser.parse(p.stdout or "{}", {}) or {}
    return obj if isinstance(obj, dict) else {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def _http_get(url: str, max_bytes: int = 32 * 1024 * 1024) -> bytes:
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if HEADERS_JSON:
        parser = JSONParser()
        hdrs = parser.parse(HEADERS_JSON, {}) or {}
        if isinstance(hdrs, dict):
            headers.update(hdrs)
    # TIMEOUTS FORBIDDEN: always disable client-side timeouts explicitly.
    with httpx.Client(timeout=None, follow_redirects=True) as cli:
        with cli.stream("GET", url, headers=headers) as r:
            out = bytearray()
            for chunk in r.iter_bytes():
                out.extend(chunk)
            return bytes(out)


def is_youtube(url: str) -> bool:
    return bool(YOUTUBE_RX.search(url))


def is_generic_media(url: str) -> bool:
    return bool(GENERIC_RX.search(url))


def is_pdf(url: str) -> bool:
    return bool(PDF_RX.search(url))


def is_image(url: str) -> bool:
    return bool(IMAGE_RX.search(url))


def save_temp(b: bytes, suffix: str) -> str:
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fd.write(b)
    fd.flush()
    fd.close()
    return fd.name


def _extract_audio_with_ffmpeg(in_url: str, out_wav: str, start=None, end=None) -> None:
    ss = ["-ss", str(start)] if start else []
    to = ["-to", str(end)] if end else []
    cmd = [
        "ffmpeg",
        "-y",
        *ss,
        *to,
        "-i",
        in_url,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        out_wav,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def resolve_media(url: str, wanted: str = "auto", ref_hint: str = "main") -> Dict[str, Any]:
    # PDFs
    if is_pdf(url):
        b = _http_get(url, max_bytes=64 * 1024 * 1024)
        path = save_temp(b, ".pdf")
        return {"type": "pdf", "meta": {"sha256": _sha256_bytes(b)}, "pdf_path": path}
    # Images
    if is_image(url):
        b = _http_get(url, max_bytes=16 * 1024 * 1024)
        ext = "." + (url.split("?")[0].split(".")[-1][:5] or "png")
        path = save_temp(b, ext)
        return {"type": "image", "meta": {"sha256": _sha256_bytes(b)}, "image_path": path}
    # YouTube/Vimeo/TikTok/Twitter/X
    if is_youtube(url) or any(s in url.lower() for s in ("vimeo.com", "tiktok.com", "twitter.com", "x.com")):
        info = _yt_dlp_info(url)
        direct = None
        if isinstance(info.get("formats"), list):
            audio_only = [
                f
                for f in info["formats"]
                if f.get("vcodec", "none") == "none" and f.get("acodec") != "none" and f.get("url")
            ]
            if audio_only:
                direct = sorted(audio_only, key=lambda f: f.get("abr", 0) or 0, reverse=True)[0]["url"]
        out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        _extract_audio_with_ffmpeg(direct or url, out_wav)
        return {
            "type": "video",
            "meta": {
                "title": info.get("title", ""),
                "desc": info.get("description", ""),
                "duration": info.get("duration") or info.get("duration_string"),
                "uploader": info.get("uploader"),
            },
            "raw_url": direct or url,
            "audio_wav": out_wav,
            "text": "",
        }
    # Generic direct media
    if is_generic_media(url):
        out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        _extract_audio_with_ffmpeg(url, out_wav)
        return {"type": "media", "raw_url": url, "audio_wav": out_wav, "text": ""}
    # Fallback HTML
    b = _http_get(url, max_bytes=8 * 1024 * 1024)
    return {"type": "html", "html": b.decode("utf-8", "replace")}


