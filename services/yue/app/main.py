from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore
import torch


MODEL_ID = os.getenv("YUE_MODEL_ID", os.getenv("MUSIC_MODEL_ID", "facebook/musicgen-medium"))

app = FastAPI(title="YuE (local) - Lyrics to Song", version="0.1.0")

_model = None
_proc = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    global _model, _proc
    if _model is None:
        dtype = torch.float16 if _device == "cuda" else torch.float32
        _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype)
        _model = _model.to(_device)
        _proc = AutoProcessor.from_pretrained(MODEL_ID)
    return _model, _proc


@app.on_event("startup")
async def _warmup():
    # Trigger model download on container start
    get_model()

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/music/song")
async def song(body: Dict[str, Any]):
    lyrics = (body.get("lyrics") or "").strip()
    style_tags = body.get("style_tags") or []
    length_s = int(body.get("duration_s") or 30)
    if not lyrics:
        return JSONResponse(status_code=400, content={"error": "missing lyrics"})
    prompt = ((", ".join(style_tags) + ": ") if style_tags else "") + lyrics
    model, proc = get_model()
    inputs = proc(text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    # Crude mapping: 50 tokens ≈ ~1s for small/medium configs
    max_new_tokens = max(50, int(length_s) * 50)
    with torch.inference_mode():
        if _device == "cuda":
            with torch.cuda.amp.autocast():
                audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=max_new_tokens)
        else:
            audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=max_new_tokens)
    audio = audio_values[0, 0].cpu().numpy().astype(np.float32)
    buf = io.BytesIO()
    sr = 32000
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    # Minimal structure stub
    structure = {"sections": [{"name": "full", "start": 0.0, "end": float(length_s)}], "bpm": body.get("bpm"), "key": body.get("key")}
    return {"audio_wav_base64": b64, "sample_rate": sr, "structure": structure, "model": f"yue:{MODEL_ID}"}


