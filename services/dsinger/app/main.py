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
from fastapi.responses import JSONResponse

app = FastAPI(title="DiffSinger (stub)", version="0.1.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "dsinger.log")
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
    logging.getLogger("dsinger.logging").info("dsinger logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("dsinger.logging").warning("dsinger file logging disabled: %s", _ex, exc_info=True)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/sing")
async def sing(body: Dict[str, Any]):
    score = body.get("score_json")
    if not score and not body.get("score_url"):
        return JSONResponse(status_code=400, content={"error": "missing score"})
    sr = 32000
    wav = np.zeros(sr * int(body.get("seconds", 4) or 4), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return {"audio_wav_base64": base64.b64encode(buf.getvalue()).decode("ascii"), "sample_rate": sr}


