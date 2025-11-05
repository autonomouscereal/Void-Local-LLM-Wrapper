from __future__ import annotations

import io
import base64
import numpy as np
import soundfile as sf
import pyloudnorm as pl
from fastapi import FastAPI
from fastapi.responses import JSONResponse


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
            y, sr = sf.read(io.BytesIO(requests.get(wav_url, timeout=30).content), always_2d=False)
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


