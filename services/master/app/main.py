from __future__ import annotations

import io
import base64
import numpy as np
import soundfile as sf
import pyloudnorm as pl
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging, sys

# ---- Logging (stdout + shared log volume file) ----
try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "master.log")
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
    logging.getLogger(__name__).info("master logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl))
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger(__name__).warning("master file logging disabled: %s", _ex, exc_info=True)


def lufs(y: np.ndarray, sr: int) -> float:
    return pl.Meter(sr).integrated_loudness(y)


def limiter(y: np.ndarray, tp_ceiling_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y) + 1e-9))
    target = 10 ** (tp_ceiling_db / 20.0)
    if peak > target:
        y = y * (target / peak)
    return y.astype(np.float32)


app = FastAPI(title="Mastering Service", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/master")
async def master(body: dict):
    try:
        wav_url = body.get("wav_url")
        wav_bytes = body.get("wav_bytes")
        lufs_target = float(body.get("lufs_target", -14.0))
        tp_ceiling_db = float(body.get("tp_ceiling_db", -1.0))
        if wav_bytes:
            data = base64.b64decode(wav_bytes)
            y, sr = sf.read(io.BytesIO(data), always_2d=False)
        elif wav_url:
            import requests
            y, sr = sf.read(io.BytesIO(requests.get(wav_url).content), always_2d=False)
        else:
            return JSONResponse(status_code=400, content={"error": "missing wav_url or wav_bytes"})
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        before = {"lufs": lufs(y, int(sr))}
        meter = pl.Meter(int(sr))
        loud = meter.integrated_loudness(y)
        y = pl.normalize.loudness(y, loud, lufs_target)
        y = limiter(y, tp_ceiling_db)
        after = {"lufs": lufs(y, int(sr))}
        buf = io.BytesIO()
        sf.write(buf, y, int(sr), format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"ok": True, "audio_wav_base64": b64, "metrics_before": before, "metrics_after": after}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


