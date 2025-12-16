from __future__ import annotations

import base64
import os
from typing import Any, Dict

import httpx  # type: ignore


class RestMusicProvider:
    """
    Minimal HTTP-based music provider used by music.infinite.windowed.

    This class is deliberately thin: it forwards compose() calls to a single
    REST endpoint exposed by the music backend (configured via MUSIC_API_URL).

    Assumptions about the backend contract (adjust as needed for your stack):
    - POST {base_url}/generate with JSON body containing:
        {
          "prompt": str,
          "seconds": int,
          "bpm": Optional[int],
          "sample_rate": int,
          "channels": int,
          "seed": Optional[int],
          "instrumental_only": Optional[bool],
          "style": Optional[str],
        }
    - On success, the backend responds with either:
        1) raw audio/wav bytes (Content-Type: audio/*), or
        2) JSON containing a base64-encoded wav payload under one of:
           result["wav_bytes_b64"] | result["wav_base64"] | result["audio_base64"]

    The provider normalizes all of these into:
        {
          "wav_bytes": <bytes>,
          "sample_rate": int,
          "channels": int,
          "model": str (optional),
          "stems": list (optional),
        }
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base = (base_url or os.getenv("MUSIC_API_URL") or "").rstrip("/")

    def compose(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self._base:
            # Never raise from tool providers; return empty audio with a structured error.
            sample_rate = int(args.get("sample_rate") or 44100)
            channels = int(args.get("channels") or 2)
            return {
                "wav_bytes": b"",
                "sample_rate": sample_rate,
                "channels": channels,
                "error": {
                    "code": "config_missing",
                    "message": "MUSIC_API_URL is not configured for RestMusicProvider",
                },
            }

        seconds = int(args.get("length_s") or 30)
        sample_rate = int(args.get("sample_rate") or 44100)
        channels = int(args.get("channels") or 2)
        payload: Dict[str, Any] = {
            "prompt": str(args.get("prompt") or ""),
            "seconds": seconds,
            "bpm": args.get("bpm"),
            "sample_rate": sample_rate,
            "channels": channels,
            "seed": args.get("seed"),
            "instrumental_only": args.get("instrumental_only"),
            "style": args.get("style"),
        }

        url = self._base + "/generate"
        with httpx.Client(timeout=None, trust_env=False) as client:  # type: ignore[arg-type]
            resp = client.post(url, json=payload)

        # Raw audio response (e.g., Content-Type: audio/wav)
        ctype = resp.headers.get("content-type") or ""
        if ctype.startswith("audio/") or ctype == "application/octet-stream":
            wav_bytes = resp.content or b""
            return {
                "wav_bytes": wav_bytes,
                "sample_rate": sample_rate,
                "channels": channels,
            }

        # JSON envelope with base64 payload
        data: Any
        try:
            data = resp.json()
        except Exception:
            data = {}

        if isinstance(data, dict):
            inner = data.get("result") if isinstance(data.get("result"), dict) else data
            b64 = (
                inner.get("wav_bytes_b64")
                or inner.get("wav_base64")
                or inner.get("audio_base64")
            )
            if isinstance(b64, str) and b64:
                try:
                    wav_bytes = base64.b64decode(b64)
                except Exception:
                    wav_bytes = b""
            else:
                wav_bytes = b""
            return {
                "wav_bytes": wav_bytes,
                "sample_rate": int(inner.get("sample_rate") or sample_rate),
                "channels": int(inner.get("channels") or channels),
                "model": inner.get("model") or "music",
                "stems": inner.get("stems") or [],
            }

        # Fallback: unknown format; return empty audio so callers can surface a structured error.
        return {
            "wav_bytes": b"",
            "sample_rate": sample_rate,
            "channels": channels,
        }


