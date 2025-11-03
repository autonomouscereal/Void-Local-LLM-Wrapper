from __future__ import annotations

import io
import os
from typing import Dict, Any

import requests
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from transformers import AutoProcessor, AutoModelForVision2Seq


MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")

app = FastAPI(title="VLM Reviewer", version="0.1.0")

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
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    model, proc = get_model()
    inputs = proc(images=img, text=prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=256)
    text = proc.batch_decode(out, skip_special_tokens=True)[0]
    return {"text": text}


