from __future__ import annotations

import base64, io
import logging
import os
import sys
from typing import Any, Dict
import numpy as np
import soundfile as sf
from fastapi import FastAPI

app = FastAPI(title="MFA Alignment (stub)", version="0.1.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "mfa.log")
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
    logging.getLogger("mfa.logging").info("mfa logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("mfa.logging").warning("mfa file logging disabled: %s", _ex, exc_info=True)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/align")
async def align(body: Dict[str, Any]):
    lyrics = (body.get("lyrics") or "").strip()
    wav_b64 = body.get("wav_bytes")
    # decode wav (unused in stub)
    if wav_b64:
        data = base64.b64decode(wav_b64)
        _y, _sr = sf.read(io.BytesIO(data), always_2d=False)
    words = [w for w in lyrics.split() if w]
    # naive timings: 300 ms per word
    align = []
    t = 0.0
    for w in words:
        t0 = t
        t1 = t0 + 0.3
        align.append({"word": w, "t0": round(t0, 3), "t1": round(t1, 3)})
        t = t1
    return {"alignment": align, "confidence": 0.7}


