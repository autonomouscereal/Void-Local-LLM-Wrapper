from __future__ import annotations

import base64
import io
import json
import logging
import os
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from TTS.api import TTS  # type: ignore


MODEL_NAME = os.getenv("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")

app = FastAPI(title="XTTS TTS Service", version="0.3.0")

# Device selection is global for this container; individual engines are per-voice.
_TTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# In-process cache: canonical_voice_key -> TTS engine
_TTS_ENGINES: Dict[str, TTS] = {}

# Optional mapping from logical voice_id to explicit model identifier.
# Example env:
#   XTTS_VOICE_MODEL_MAP='{"lead":"tts_models/.../xtts_v2","narrator":"/models/xtts_narrator"}'
_TTS_MODEL_MAP: Dict[str, str] = {}


def _load_voice_model_map() -> None:
    """
    Initialize _TTS_MODEL_MAP from XTTS_VOICE_MODEL_MAP when present.

    The mapping is always best-effort and never fatal; malformed JSON falls
    back to an empty map and logs a warning.
    """
    global _TTS_MODEL_MAP
    raw = os.getenv("XTTS_VOICE_MODEL_MAP", "").strip()
    if not raw:
        _TTS_MODEL_MAP = {}
        return
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            _TTS_MODEL_MAP = {str(k): str(v) for k, v in obj.items()}
            logging.info("xtts.voice_model_map.loaded keys=%s", sorted(_TTS_MODEL_MAP.keys()))
        else:
            logging.warning("xtts.voice_model_map.invalid_type type=%s", type(obj))
            _TTS_MODEL_MAP = {}
    except Exception:
        logging.exception("xtts.voice_model_map.parse_error")
        _TTS_MODEL_MAP = {}


_load_voice_model_map()


def _canonical_voice_key(voice_id: Optional[str]) -> str:
    """
    Map the incoming logical voice_id to a canonical cache key.

    - If voice_id matches an explicit entry in _TTS_MODEL_MAP, use it directly.
    - Otherwise, fall back to a shared 'default' engine bound to MODEL_NAME.
    """
    if isinstance(voice_id, str):
        vid = voice_id.strip()
        if vid and vid in _TTS_MODEL_MAP:
            return vid
    return "default"


def _resolve_model_identifier(voice_key: str) -> str:
    """
    Resolve the concrete model identifier for a canonical voice key.

    When voice_key == 'default' (or unknown), use MODEL_NAME so we have a
    single shared engine. When voice_key matches XTTS_VOICE_MODEL_MAP, use
    that entry so each logical voice can be backed by its own model.
    """
    if voice_key != "default" and voice_key in _TTS_MODEL_MAP:
        return _TTS_MODEL_MAP[voice_key]
    return MODEL_NAME


def _get_engine_for_voice(voice_id: Optional[str]) -> Tuple[TTS, str, str, str]:
    """
    Look up or create a TTS engine for the given logical voice_id.

    Returns (engine, canonical_voice_key, model_identifier, lifecycle),
    where lifecycle is 'created' or 'reused' for logging.
    """
    global _TTS_ENGINES

    voice_key = _canonical_voice_key(voice_id)
    if voice_key in _TTS_ENGINES:
        model_identifier = _resolve_model_identifier(voice_key)
        logging.info(
            "tts.engine.reused voice_id=%s voice_key=%s model=%s device=%s",
            voice_id or "",
            voice_key,
            model_identifier,
            _TTS_DEVICE,
        )
        return _TTS_ENGINES[voice_key], voice_key, model_identifier, "reused"

    model_identifier = _resolve_model_identifier(voice_key)
    logging.info(
        "tts.engine.create.start voice_id=%s voice_key=%s model=%s device=%s",
        voice_id or "",
        voice_key,
        model_identifier,
        _TTS_DEVICE,
    )
    engine = TTS(model_identifier)  # may download/initialize; failures should surface loudly
    if _TTS_DEVICE.startswith("cuda"):
        # Do not swallow device errors here; if CUDA is misconfigured, fail fast.
        engine.to(_TTS_DEVICE)  # type: ignore[arg-type]
    _TTS_ENGINES[voice_key] = engine
    logging.info(
        "tts.engine.created voice_id=%s voice_key=%s model=%s device=%s",
        voice_id or "",
        voice_key,
        model_identifier,
        _TTS_DEVICE,
    )
    return engine, voice_key, model_identifier, "created"


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/tts")
async def tts(body: Dict[str, Any]):
    trace_id = body.get("trace_id") if isinstance(body.get("trace_id"), str) else "tt_xtts_unknown"
    segment_id = (body.get("segment_id") or "").strip() if isinstance(body.get("segment_id"), str) else ""
    text_raw = body.get("text") or ""
    text = text_raw.strip()
    language = body.get("language") or "en"
    # Logical voice identifier from upstream (voice_id preferred, voice as fallback).
    raw_voice_id = body.get("voice_id") or body.get("voice") or ""
    voice_id = raw_voice_id.strip() if isinstance(raw_voice_id, str) else ""
    if not voice_id:
        # Enforce explicit voice selection so upstream callers wire a concrete
        # speaker profile; this avoids opaque RuntimeError messages like
        # "Neither `speaker_wav` nor `speaker_id` was specified".
        return JSONResponse(
            status_code=200,
            content={
                "schema_version": 1,
                "trace_id": trace_id,
                "ok": False,
                "result": None,
                "error": {
                    "code": "xtts_missing_voice",
                    "message": "No voice_id/voice provided for XTTS request.",
                    "stack": "".join(traceback.format_stack()),
                },
            },
        )
    if not text:
        # Always return a canonical envelope; do not rely on HTTP status codes or
        # raised exceptions for control flow. Include a synthetic stack trace so
        # upstream callers can debug even validation failures.
        return JSONResponse(
            status_code=200,
            content={
                "schema_version": 1,
                "trace_id": trace_id,
                "ok": False,
                "result": None,
                "error": {
                    "code": "ValidationError",
                    "message": "Missing 'text' for TTS.",
                    "stack": "".join(traceback.format_stack()),
                },
            },
        )

    try:
        # Per-voice engine lookup and creation with GPU preference when available.
        engine, voice_key, model_identifier, lifecycle = _get_engine_for_voice(voice_id or None)
        logging.info(
            "tts.request trace_id=%s voice_id=%s voice_key=%s model=%s device=%s engine_lifecycle=%s",
            trace_id,
            voice_id or "",
            voice_key,
            model_identifier,
            _TTS_DEVICE,
            lifecycle,
        )
        # Optional: seed for determinism when provided.
        seed = body.get("seed")
        if isinstance(seed, int):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        # XTTS API supports language + speaker selection; interpret the logical
        # voice_id as a speaker_id so multi-voice setups can route to distinct
        # embeddings or base voices. Any mismatch or missing speaker mapping
        # will be surfaced via the structured error envelope below.
        wav = engine.tts(text=text, speaker_id=voice_id, language=language)  # type: ignore[call-arg]
        wav_arr = np.array(wav, dtype=np.float32)
        buf = io.BytesIO()
        sample_rate = 22050
        sf.write(buf, wav_arr, sample_rate, format="WAV")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return JSONResponse(
            status_code=200,
            content={
                "schema_version": 1,
                "trace_id": trace_id,
                "ok": True,
                "result": {
                    "audio_wav_base64": b64,
                    "sample_rate": sample_rate,
                    "language": language,
                    "voice_id": voice_id,
                    "xtts_model_key": voice_key,
                    # Optional segment identifier from upstream orchestrator; echoed back
                    # so downstream RVC/mixer code can attribute this chunk.
                    "segment_id": segment_id or None,
                    # Derived duration in seconds; XTTS core may not return this directly.
                    "duration_s": float(len(wav_arr)) / float(sample_rate) if sample_rate > 0 else 0.0,
                    "model": model_identifier,
                },
                "error": None,
            },
        )
    except Exception as ex:
        logging.exception("xtts.tts.error trace_id=%s", trace_id)
        # Surface all internal failures as a structured envelope instead of
        # letting exceptions or HTTP 500s leak to the caller.
        return JSONResponse(
            status_code=200,
            content={
                "schema_version": 1,
                "trace_id": trace_id,
                "ok": False,
                "result": None,
                "error": {
                    "code": ex.__class__.__name__ or "InternalError",
                    "message": str(ex),
                    "stack": traceback.format_exc(),
                },
            },
        )

