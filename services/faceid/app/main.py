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

# Canonical InsightFace antelopev2 path (shared with ComfyUI via comfyui_models volume).
_INSIGHTFACE_ROOT = os.environ.get("INSIGHTFACE_MODEL_ROOT", "/opt/models/insightface")
_ANTELOPE_DIR = os.path.join(_INSIGHTFACE_ROOT, "models", "antelopev2")
_REQUIRED_FILES = ("glintr100.onnx", "scrfd_10g_bnkps.onnx")


def _check_antelopev2_present() -> None:
    missing = []
    for name in _REQUIRED_FILES:
        path = os.path.join(_ANTELOPE_DIR, name)
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            missing.append(path)

def get_model():
    global _model
    if _model is None:
        _check_antelopev2_present()
        # Use antelopev2 pack (glintr100 + scrfd_10g_bnkps) from the shared models root.
        ctx_id = int(os.environ.get("FACEID_CTX_ID", "0"))  # 0 = first GPU, -1 = CPU
        _model = insightface.app.FaceAnalysis(name="antelopev2", root=_INSIGHTFACE_ROOT)
        _model.prepare(ctx_id=ctx_id)
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
    r = requests.get(image_url)
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


