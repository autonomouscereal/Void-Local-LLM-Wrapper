from __future__ import annotations

import base64
import io
import logging
import os
import sys
from typing import Dict, Any
import numpy as np
import soundfile as sf
from fastapi import FastAPI

app = FastAPI(title="SFX (stub)", version="0.1.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "sfx.log")
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
    logging.getLogger(__name__).info("sfx logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger(__name__).warning("sfx file logging disabled: %s", _ex, exc_info=True)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


def _silent_wav(seconds: int = 2, sr: int = 32000) -> bytes:
    import numpy as _np
    buf = io.BytesIO()
    sf.write(buf, _np.zeros(sr * max(1, seconds), dtype=_np.float32), sr, format="WAV")
    return buf.getvalue()


@app.post("/sfx")
async def sfx(body: Dict[str, Any]):
    return {"audio_wav_base64": base64.b64encode(_silent_wav()).decode("ascii"), "sample_rate": 32000}


@app.post("/variation")
async def variation(body: Dict[str, Any]):
    return {"items": [{"audio_wav_base64": base64.b64encode(_silent_wav()).decode("ascii"), "score": 0.0}]}


