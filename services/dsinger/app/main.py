from __future__ import annotations

import base64
import io
from typing import Dict, Any
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="DiffSinger (stub)", version="0.1.0")


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


