from __future__ import annotations

import io
import os
from typing import Dict, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    import insightface
except Exception:
    insightface = None

app = FastAPI(title="FaceID Service", version="0.1.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if insightface is None:
            raise RuntimeError("insightface not available")
        _model = insightface.app.FaceAnalysis(name="buffalo_l")
        _model.prepare(ctx_id=-1)  # CPU
    return _model


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/embed")
async def embed(body: Dict[str, Any]):
    image_url = body.get("image_url")
    if not image_url:
        return JSONResponse(status_code=400, content={"error": "missing image_url"})
    import requests
    r = requests.get(image_url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img = np.array(img)
    model = get_model()
    faces = model.get(img)
    if not faces:
        return {"embeddings": []}
    # return first embedding for simplicity
    emb = faces[0].normed_embedding.tolist()
    return {"embeddings": [emb]}


