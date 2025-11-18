from __future__ import annotations

import io
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import soundfile as sf
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from TTS.api import TTS  # type: ignore
import torch
import base64


MODEL_NAME = os.getenv("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")

app = FastAPI(title="XTTS TTS Service", version="0.2.0")

_XTTS_ENGINE: Optional[TTS] = None
_XTTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_XTTS_BASE_SPEAKERS: List[str] = []
_XTTS_SPEAKER_MAP: Dict[str, str] = {}


def _init_xtts_base_speakers(engine: TTS) -> None:
    """
    Discover available XTTS speakers once at startup and store them in
    _XTTS_BASE_SPEAKERS. This ensures we only ever pass valid speaker IDs to
    xtts.tts, so FileNotFoundError for voice files cannot happen.
    """
    global _XTTS_BASE_SPEAKERS

    speaker_manager = getattr(engine.tts_model, "speaker_manager", None)
    if speaker_manager is not None and getattr(speaker_manager, "speakers", None):
        names = list(speaker_manager.speakers.keys())
        names.sort()
        _XTTS_BASE_SPEAKERS = names
        logging.info("xtts.speakers.discovered via speaker_manager: %s", _XTTS_BASE_SPEAKERS)
        return

    voices_root = Path(
        os.getenv(
            "XTTS_VOICES_DIR",
            Path.home()
            / ".local"
            / "share"
            / "tts"
            / "tts_models--multilingual--multi-dataset--xtts_v2"
            / "voices",
        )
    )
    if voices_root.exists():
        names: List[str] = []
        for path in voices_root.glob("*.pth"):
            if path.is_file():
                names.append(path.stem)
        names.sort()
        _XTTS_BASE_SPEAKERS = names
        logging.info("xtts.speakers.discovered via voices_dir=%s: %s", str(voices_root), _XTTS_BASE_SPEAKERS)
    else:
        logging.warning("xtts.voices_dir_missing path=%s", str(voices_root))
        _XTTS_BASE_SPEAKERS = []


def get_xtts_engine() -> TTS:
    """
    Lazily construct the XTTS engine and discover base speakers. The engine
    object never leaves this module.
    """
    global _XTTS_ENGINE
    if _XTTS_ENGINE is None:
        logging.info("xtts.load.start model=%s device=%s", MODEL_NAME, _XTTS_DEVICE)
        engine = TTS(MODEL_NAME)  # may download/initialize; failures should surface loudly
        if _XTTS_DEVICE.startswith("cuda"):
            # Do not swallow device errors here; if CUDA is misconfigured, fail fast.
            engine.to("cuda")  # type: ignore[arg-type]
        _init_xtts_base_speakers(engine)
        _XTTS_ENGINE = engine
        logging.info("xtts.load.done model=%s", MODEL_NAME)
    return _XTTS_ENGINE  # type: ignore[return-value]


def map_voice_to_xtts_speaker(voice_id: str) -> Optional[str]:
    """
    Deterministically map a logical voice identifier (voice_id or voice name)
    to one of the available XTTS base speakers. If there are fewer XTTS
    speakers than distinct voice_ids, reuse them round-robin.

    Returns None when no base speakers are available; the caller must then
    omit the 'speaker' argument entirely.
    """
    global _XTTS_SPEAKER_MAP

    if not isinstance(voice_id, str) or not voice_id.strip():
        return None

    if not _XTTS_BASE_SPEAKERS:
        # No explicit speakers discovered; rely on XTTS internal defaults.
        return None

    if voice_id in _XTTS_SPEAKER_MAP:
        return _XTTS_SPEAKER_MAP[voice_id]

    index = len(_XTTS_SPEAKER_MAP) % len(_XTTS_BASE_SPEAKERS)
    speaker_id = _XTTS_BASE_SPEAKERS[index]
    _XTTS_SPEAKER_MAP[voice_id] = speaker_id
    logging.info("xtts.speaker_mapping voice_id=%s base_speaker=%s", voice_id, speaker_id)
    return speaker_id


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
    voice_id = (body.get("voice_id") or body.get("voice") or "").strip()
    if not text:
        return JSONResponse(
            status_code=400,
            content={
                "schema_version": 1,
                "trace_id": trace_id,
                "ok": False,
                "result": None,
                "error": {"code": "bad_request", "message": "Missing 'text' for TTS."},
            },
        )
    engine = get_xtts_engine()
    xtts_speaker = map_voice_to_xtts_speaker(voice_id) if voice_id else None
    if xtts_speaker:
        wav = engine.tts(text=text, speaker=xtts_speaker, language=language)  # type: ignore[call-arg]
    else:
        # No explicit base speakers available; rely on XTTS internal default speaker.
        wav = engine.tts(text=text, language=language)  # type: ignore[call-arg]
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
                "xtts_speaker": xtts_speaker,
                # Optional segment identifier from upstream orchestrator; echoed back
                # so downstream RVC/mixer code can attribute this chunk.
                "segment_id": segment_id or None,
                # Derived duration in seconds; XTTS core may not return this directly.
                "duration_s": float(len(wav_arr)) / float(sample_rate) if sample_rate > 0 else 0.0,
                "model": MODEL_NAME,
            },
            "error": None,
        },
    )

