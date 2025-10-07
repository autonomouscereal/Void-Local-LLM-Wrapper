from __future__ import annotations

import os
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel


MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large-v3")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "8"))

app = FastAPI(title="Whisper ASR Service", version="0.1.0")

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
    text = " ".join([seg.text for seg in segments])
    return {"text": text, "language": info.language}


