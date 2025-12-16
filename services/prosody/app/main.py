from __future__ import annotations

from typing import Any, Dict
import logging
import os
import sys
from fastapi import FastAPI

app = FastAPI(title="Prosody Predictor (stub)", version="0.1.0")

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "prosody.log")
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
    logging.getLogger(__name__).info("prosody logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger(__name__).warning("prosody file logging disabled: %s", _ex, exc_info=True)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/suggest")
async def suggest(body: Dict[str, Any]):
    score = body.get("score_json") or {}
    notes = score.get("notes") or []
    # naive: stretch long vowels by +5%, tiny pitch nudge +10 cents on every 4th note
    stretch = []
    pitch = []
    for i, _ in enumerate(notes):
        if i % 4 == 0:
            pitch.append({"note_idx": i, "cents": 10})
        if i % 3 == 0:
            stretch.append({"note_idx": i, "ratio": 1.05})
    return {"note_stretch": stretch[:16], "pitch_nudge_cents": pitch[:16]}


