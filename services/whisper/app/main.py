from __future__ import annotations

import os
import logging
import sys
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel


MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large-v3")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "8"))

app = FastAPI(title="Whisper ASR Service", version="0.1.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "whisper.log")
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
    logging.getLogger("whisper.logging").info("whisper logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("whisper.logging").warning("whisper file logging disabled: %s", _ex, exc_info=True)

_model = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    return _model


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(body: Dict[str, Any]):
    audio_url = body.get("audio_url")
    language = body.get("language")
    if not audio_url:
        return JSONResponse(status_code=400, content={"error": "missing audio_url"})
    model = get_model()
    segments, info = model.transcribe(audio_url, beam_size=1, language=language)
    
    # Convert generator to list and build response segments
    seg_list = []
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)
        seg_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "speaker_label": "SPEAKER_00" # Placeholder until diarization model is integrated
        })
    
    text = " ".join(text_parts)
    return {
        "text": text,
        "language": info.language,
        "segments": seg_list
    }


