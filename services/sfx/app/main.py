from __future__ import annotations

import base64
import io
from typing import Dict, Any
import numpy as np
import soundfile as sf
from fastapi import FastAPI

app = FastAPI(title="SFX (stub)", version="0.1.0")


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


