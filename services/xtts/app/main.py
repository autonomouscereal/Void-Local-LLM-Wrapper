from __future__ import annotations

import base64
import io
import logging
import os
import traceback
import sys
from typing import Any, Dict, Optional, Tuple

import json  # kept only for internal dumps; parsing goes via JSONParser
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from TTS.api import TTS  # type: ignore
from void_json.json_parser import JSONParser


MODEL_NAME = os.getenv("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")

app = FastAPI(title="XTTS TTS Service", version="0.3.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "xtts.log")
    _lvl = getattr(logging, (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_lvl,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(_log_file, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger("xtts.logging").info("xtts logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("xtts.logging").warning("xtts file logging disabled: %s", _ex, exc_info=True)

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
    # Primary source (if provided): environment variable.
    raw = os.getenv("XTTS_VOICE_MODEL_MAP", "").strip()
    map_path = os.getenv(
        "XTTS_VOICE_MODEL_MAP_PATH",
        # Lives under the tts_cache volume in docker-compose so it persists.
        "/root/.local/share/tts/voice_model_map.json",
    )
    os.makedirs(os.path.dirname(map_path), exist_ok=True)

    if raw:
        # Backwards-compatible: allow JSON from env, then persist it to disk
        # so subsequent restarts don't rely on the env being present.
        parser = JSONParser()
        obj = parser.parse(raw, {}) or {}
        if isinstance(obj, dict):
            _TTS_MODEL_MAP = {str(k): str(v) for k, v in obj.items()}
            logging.info(
                "xtts.voice_model_map.loaded_from_env keys=%s",
                sorted(_TTS_MODEL_MAP.keys()),
            )
            tmp = map_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    _TTS_MODEL_MAP,
                    f,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            os.replace(tmp, map_path)
            return
        else:
            logging.warning("xtts.voice_model_map.invalid_type type=%s", type(obj))
            _TTS_MODEL_MAP = {}
            return

    # Fallback: load from disk if present (zero-config path, backed by volume).
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _TTS_MODEL_MAP = {str(k): str(v) for k, v in data.items()}
            logging.info(
                "xtts.voice_model_map.loaded_from_disk path=%s keys=%s",
                map_path,
                sorted(_TTS_MODEL_MAP.keys()),
            )
        else:
            logging.warning(
                "xtts.voice_model_map.disk_invalid_type path=%s type=%s",
                map_path,
                type(data),
            )
            _TTS_MODEL_MAP = {}
    else:
        # No env and no disk map: start with an empty mapping and log once.
        _TTS_MODEL_MAP = {}
        logging.info(
            "xtts.voice_model_map.empty_using_default path=%s", map_path
        )


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

