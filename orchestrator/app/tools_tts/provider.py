from __future__ import annotations

from typing import Dict, Any
import os
import traceback

import httpx as _hxsync  # type: ignore

from ..json_parser import JSONParser
from ..tools import emit_progress
from void_envelopes import _build_success_envelope, _build_error_envelope


class _TTSProvider:
    """
    Shared XTTS provider used by tts.speak and Film2 vocal rendering.
    Adapts the external XTTS /tts response into the canonical envelope that
    run_tts_speak expects.
    """

    def __init__(self, base_url: str | None = None) -> None:
        url = base_url if isinstance(base_url, str) and base_url.strip() else os.getenv("XTTS_API_URL", "")
        self._base_url = url.rstrip("/") if isinstance(url, str) else ""

    def speak(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = payload.get("trace_id") if isinstance(payload.get("trace_id"), str) else "tt_xtts_unknown"
        emit_progress({"stage": "request", "target": "xtts"})
        # Ensure language is always set; default to English if absent.
        lang = payload.get("language")
        if not isinstance(lang, str) or not lang.strip():
            payload["language"] = "en"
        base = self._base_url
        if not base:
            return _build_error_envelope(
                "xtts_unconfigured",
                "XTTS_API_URL is not configured for TTS provider.",
                trace_id,
                status=500,
                details={"stack": "".join(traceback.format_stack())},
            )
        with _hxsync.Client(timeout=None, trust_env=False) as client:
            r = client.post(base + "/tts", json=payload)
            status = r.status_code
            ct = r.headers.get("content-type", "")
            if 200 <= status < 300:
                # XTTS service returns a canonical envelope: decode audio and adapt to internal shape.
                if "application/json" in ct:
                    parser = JSONParser()
                    env = parser.parse_superset(
                        r.text or "",
                        {
                            "schema_version": int,
                            "trace_id": str,
                            "ok": bool,
                            "result": dict,
                            "error": dict,
                        },
                    )
                else:
                    env = {
                        "ok": False,
                        "error": {
                            "code": "tts_invalid_body",
                            "status": status,
                            "message": "XTTS /tts returned non-JSON body",
                            "raw": r.text,
                            "stack": "".join(traceback.format_stack()),
                        },
                    }
                if isinstance(env, dict) and env.get("ok") is True:
                    inner = env.get("result") if isinstance(env.get("result"), dict) else {}
                    b64 = inner.get("audio_wav_base64") or inner.get("wav_b64")
                    wav = b""
                    if isinstance(b64, str):
                        import base64 as _b64local

                        wav = _b64local.b64decode(b64)
                    # Adapt to canonical internal envelope expected by run_tts_speak.
                    sample_rate = int(inner.get("sample_rate") or payload.get("sample_rate") or 22050)
                    language = inner.get("language") or payload.get("language") or "en"
                    voice_id = inner.get("voice_id") or payload.get("voice_id") or payload.get("voice")
                    xtts_speaker = inner.get("xtts_speaker")
                    segment_id = inner.get("segment_id") or payload.get("segment_id")
                    result = {
                        "wav_bytes": wav,
                        "duration_s": float(inner.get("duration_s") or 0.0),
                        "model": inner.get("model") or "xtts",
                        "sample_rate": sample_rate,
                        "language": language,
                        "voice_id": voice_id,
                        "voice": payload.get("voice"),
                        "xtts_speaker": xtts_speaker,
                        "segment_id": segment_id,
                    }
                    return _build_success_envelope(result, trace_id)
                # Any non-ok env from XTTS is wrapped as a tool error.
                return _build_error_envelope(
                    "tts_invalid_envelope",
                    "XTTS /tts returned non-ok envelope",
                    trace_id,
                    status=status,
                    details={"raw": env, "stack": "".join(traceback.format_stack())},
                )
            # Non-2xx status: construct error envelope.
            raw_body = r.text
            data = None
            if "application/json" in ct:
                # Best-effort parse of error bodies from XTTS; we treat the
                # whole object as an untyped mapping here.
                parser = JSONParser()
                data = parser.parse_superset(raw_body or "", dict)["coerced"]
            return _build_error_envelope(
                "tts_http_error",
                "XTTS /tts returned non-2xx or invalid body",
                trace_id,
                status=status,
                details={
                    "raw": data if data is not None else {"body": raw_body},
                    "stack": "".join(traceback.format_stack()),
                },
            )


