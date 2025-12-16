from __future__ import annotations

from typing import Dict, Any
import logging
import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="MelodyWriter", version="0.1.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "melody.log")
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
    logging.getLogger(__name__).info("melody logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger(__name__).warning("melody file logging disabled: %s", _ex, exc_info=True)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/score")
async def score(body: Dict[str, Any]):
    lyrics = (body.get("lyrics") or "").strip()
    if not lyrics:
        return JSONResponse(status_code=400, content={"error": "missing lyrics"})
    resp = {
        "bpm": int(body.get("bpm") or 140),
        "key": str(body.get("key") or "E minor"),
        "notes": [{"t0": 0.0, "t1": 0.5, "hz": 220.0, "phon": "AH0"}],
        "sections": [{"name": "verse", "t0": 0.0, "t1": 16.0}],
        "phonemes": ["AH0"],
        "words": lyrics.split(),
    }
    return resp


