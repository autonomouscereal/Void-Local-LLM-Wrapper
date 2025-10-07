from __future__ import annotations

import io
import os
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from transformers import AutoProcessor, MusicgenForConditionalGeneration


MODEL_ID = os.getenv("MUSIC_MODEL_ID", "facebook/musicgen-small")

app = FastAPI(title="MusicGen Service", version="0.1.0")

_model = None
_proc = None


def get_model():
    global _model, _proc
    if _model is None:
        _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
        _proc = AutoProcessor.from_pretrained(MODEL_ID)
    return _model, _proc


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/generate")
async def generate(body: Dict[str, Any]):
    prompt = body.get("prompt") or ""
    duration = int(body.get("duration", 8))
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "missing prompt"})
    model, proc = get_model()
    inputs = proc(text=[prompt], padding=True, return_tensors="pt")
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=duration * 50)
    audio = audio_values[0, 0].cpu().numpy()
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, 32000, format="WAV")
    buf.seek(0)
    import base64
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"audio_wav_base64": b64, "sample_rate": 32000}


