from __future__ import annotations

import io
import os
import logging
import sys
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

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "faceid.log")
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
    logging.getLogger("faceid.logging").info("faceid logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("faceid.logging").warning("faceid file logging disabled: %s", _ex, exc_info=True)

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


