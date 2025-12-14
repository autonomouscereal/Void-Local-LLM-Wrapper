from __future__ import annotations

import os
import base64
import mimetypes
from typing import Dict, Any, List
from ..json_parser import JSONParser


TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml", ".html", ".htm", ".toml", ".ini", ".cfg", ".log"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".avif", ".ico"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg", ".m4v"}
DOC_EXTS = {".pdf", ".doc", ".docx", ".rtf", ".odt"}


def _read_text(path: str, max_bytes: int = 300_000) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read(max_bytes)
        try:
            return b.decode("utf-8")
        except Exception:
            return b.decode("utf-8", "ignore")
    except Exception:
        return ""


def _b64_of(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read(max_bytes)
        return base64.b64encode(b).decode("ascii")
    except Exception:
        return ""


def ingest_file(path: str, vlm_url: str | None = None, whisper_url: str | None = None, ocr_url: str | None = None) -> Dict[str, Any]:
    """
    Returns {"texts": [str...], "meta": {...}}
    """
    texts: List[str] = []
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTS:
        txt = _read_text(path)
        if txt:
            texts.append(txt)
    elif ext in DOC_EXTS:
        # Prefer OCR for PDFs and scanned docs when service available
        if ocr_url:
            try:
                import httpx
                b64 = _b64_of(path, max_bytes=10_000_000)
                # TIMEOUTS FORBIDDEN: never pass timeout to HTTP clients
                with httpx.Client(timeout=None, trust_env=False) as client:
                    r = client.post(ocr_url.rstrip("/") + "/ocr", json={"b64": b64, "ext": ext})
                    parser = JSONParser()
                    sup = parser.parse_superset(r.text or "", {"text": str})
                    js = sup["coerced"]
                    txt = (js.get("text") or "").strip() if isinstance(js, dict) else ""
                    if txt:
                        texts.append(txt)
            except Exception:
                pass
        if not texts:
            txt = _read_text(path)
            if txt and len(txt.split()) >= 5:
                texts.append(txt)
            else:
                texts.append(f"[doc] {os.path.basename(path)} {ext}")
    elif ext in IMAGE_EXTS:
        if vlm_url:
            try:
                import httpx
                b64 = _b64_of(path)
                # TIMEOUTS FORBIDDEN
                with httpx.Client(timeout=None, trust_env=False) as client:
                    r = client.post(vlm_url.rstrip("/") + "/analyze", json={"b64": b64})
                    parser = JSONParser()
                    sup = parser.parse_superset(r.text or "", {"caption": str, "text": str})
                    js = sup["coerced"]
                    cap = (js.get("caption") or js.get("text") or "").strip() if isinstance(js, dict) else ""
                    if cap:
                        texts.append(f"[image] {os.path.basename(path)}\n{cap}")
            except Exception:
                texts.append(f"[image] {os.path.basename(path)}")
        if ocr_url:
            try:
                import httpx
                b64 = _b64_of(path)
                # TIMEOUTS FORBIDDEN
                with httpx.Client(timeout=None, trust_env=False) as client:
                    r = client.post(ocr_url.rstrip("/") + "/ocr", json={"b64": b64, "ext": ext})
                    parser = JSONParser()
                    sup = parser.parse_superset(r.text or "", {"text": str})
                    js = sup["coerced"]
                    tx = (js.get("text") or "").strip() if isinstance(js, dict) else ""
                    if tx:
                        texts.append(f"[ocr] {tx}")
            except Exception:
                pass
        if not texts:
            texts.append(f"[image] {os.path.basename(path)}")
    elif ext in AUDIO_EXTS:
        if whisper_url:
            try:
                import httpx
                b64 = _b64_of(path)
                # TIMEOUTS FORBIDDEN
                with httpx.Client(timeout=None, trust_env=False) as client:
                    r = client.post(whisper_url.rstrip("/") + "/transcribe", json={"b64": b64})
                    parser = JSONParser()
                    sup = parser.parse_superset(r.text or "", {"text": str, "transcript": str})
                    js = sup["coerced"]
                    tx = (js.get("text") or js.get("transcript") or "").strip() if isinstance(js, dict) else ""
                    if tx:
                        texts.append(f"[audio] {os.path.basename(path)}\n{tx}")
            except Exception:
                texts.append(f"[audio] {os.path.basename(path)}")
        else:
            texts.append(f"[audio] {os.path.basename(path)}")
    elif ext in VIDEO_EXTS:
        texts.append(f"[video] {os.path.basename(path)} {ext}")
    else:
        # Generic fallback: best-effort text read or stub
        txt = _read_text(path)
        if txt and len(txt.split()) >= 5:
            texts.append(txt)
        else:
            texts.append(f"[file] {os.path.basename(path)} {ext}")
    return {"texts": texts, "meta": {"ext": ext, "mime": mimetypes.guess_type(path)[0] or "application/octet-stream"}}


