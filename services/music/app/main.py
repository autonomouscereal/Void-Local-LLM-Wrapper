from __future__ import annotations

import io
import os
import base64
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s:%(name)s:%(message)s")

from .music_engine import load_music_engine, generate_music


# Primary music model directory (generic).
MUSIC_MODEL_DIR = os.getenv("MUSIC_MODEL_DIR", "/opt/models/music")

app = FastAPI(title="Music Service", version="0.2.0")

_model = None
_device = None


# --- Device selection (per script, no helpers, no gating) ---
import sys as _sys
try:
    import torch as _torch  # local to this file
    DEVICE = "cuda:0" if _torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        print("[warn] music: CUDA not available; using CPU", file=_sys.stderr, flush=True)
except Exception as _e:
    DEVICE = "cpu"
    print(f"[warn] music: torch import failed ({_e}); using CPU", file=_sys.stderr, flush=True)
USE_FP16 = (os.environ.get("USE_FP16", "0") == "1") and DEVICE.startswith("cuda")
# --- end device selection ---


def get_model():
    global _model, _device
    if _model is None:
        _device = DEVICE
        # load_music_engine initializes the global processor and returns the model.
        _model = load_music_engine(MUSIC_MODEL_DIR, device=_device)
    # Absolute guardrail: never let a tuple propagate as the model; unwrap here.
    if isinstance(_model, (tuple, list)):
        logging.info("music.get_model: unwrapping tuple model %s", type(_model))
        try:
            _model = _model[0]
        except Exception:
            # If this ever happens, surface via the generate() error envelope.
            pass
    return _model, _device


@app.get("/healthz")
async def healthz():
    ok = os.path.isdir(MUSIC_MODEL_DIR) and bool(os.listdir(MUSIC_MODEL_DIR))
    return {"status": "ok" if ok else "missing_music_model", "model_dir": MUSIC_MODEL_DIR}


@app.post("/generate")
async def generate(body: Dict[str, Any]):
    prompt = body.get("prompt") or ""
    seconds = int(body.get("seconds", 8))
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "missing prompt"})
    try:
        model, device = get_model()
        wav = generate_music(
            model,
            prompt=prompt,
            seconds=seconds,
            seed=body.get("seed"),
            refs=body.get("refs"),
            device=device,
        )
        if not isinstance(wav, (bytes, bytearray)):
            # Expecting raw WAV bytes; if ndarray, encode to wav
            import soundfile as sf

            buf = io.BytesIO()
            arr = np.asarray(wav, dtype=np.float32)
            sf.write(buf, arr, 32000, format="WAV")
            wav = buf.getvalue()
        b64 = base64.b64encode(wav).decode("utf-8")
        return {"audio_wav_base64": b64, "sample_rate": 32000}
    except Exception as ex:  # surface full error as structured JSON
        import traceback

        logging.exception("music.generate error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "music_internal_error",
                    "message": str(ex),
                    "traceback": traceback.format_exc(),
                }
            },
        )


