from __future__ import annotations

import base64, io
from typing import Any, Dict
import numpy as np
import soundfile as sf
from fastapi import FastAPI

app = FastAPI(title="MFA Alignment (stub)", version="0.1.0")


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


