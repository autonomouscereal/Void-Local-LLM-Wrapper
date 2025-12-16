from __future__ import annotations

import base64
import io
import os
import logging
import sys
from typing import Any, Dict

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, MusicgenForConditionalGeneration  # type: ignore
import torch


MODEL_ID = os.getenv("SAO_MODEL_ID", "facebook/musicgen-small")

app = FastAPI(title="Stable Audio Open (local) - Timed", version="0.1.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "sao.log")
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
    logging.getLogger("sao.logging").info("sao logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("sao.logging").warning("sao file logging disabled: %s", _ex, exc_info=True)

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
    get_model()

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/music/timed")
async def timed(body: Dict[str, Any]):
    text = (body.get("text") or "").strip()
    seconds = int(body.get("seconds") or 8)
    if not text:
        return JSONResponse(status_code=400, content={"error": "missing text"})
    model, proc = get_model()
    inputs = proc(text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    max_new_tokens = max(50, seconds * 50)
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
    return {"audio_wav_base64": b64, "sample_rate": sr, "model": f"sao:{MODEL_ID}", "seconds": seconds}


