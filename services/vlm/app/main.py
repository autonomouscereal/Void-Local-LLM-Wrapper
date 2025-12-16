from __future__ import annotations

import io
import os
import logging
import sys
from typing import Dict, Any

import requests
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from transformers import AutoProcessor, AutoModelForVision2Seq


MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")

app = FastAPI(title="VLM Reviewer", version="0.1.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "vlm.log")
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
    logging.getLogger(__name__).info("vlm logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger(__name__).warning("vlm file logging disabled: %s", _ex, exc_info=True)

_model = None
_proc = None


def get_model():
    global _model, _proc
    if _model is None:
        _model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="auto")
        _proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return _model, _proc


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(body: Dict[str, Any]):
    image_url = body.get("image_url")
    prompt = body.get("prompt") or "Describe this image."
    if not image_url:
        return JSONResponse(status_code=400, content={"error": "missing image_url"})
    r = requests.get(image_url)
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    model, proc = get_model()
    inputs = proc(images=img, text=prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=256)
    text = proc.batch_decode(out, skip_special_tokens=True)[0]
    return {"text": text}


