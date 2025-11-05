from __future__ import annotations

import io
import os
from typing import Dict, Any

import soundfile as sf
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from TTS.api import TTS
except Exception:
    TTS = None
import torch


MODEL_NAME = os.getenv("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")

app = FastAPI(title="XTTS TTS Service", version="0.1.0")

_tts = None


def get_tts():
    global _tts
    if _tts is None:
        if TTS is None:
            raise RuntimeError("coqui-tts not available")
        use_cuda = torch.cuda.is_available()
        try:
            _tts = TTS(MODEL_NAME)
            if use_cuda:
                # Some TTS models expose .to(); ignore if unsupported
                try:
                    _tts.to('cuda')  # type: ignore
                except Exception:
                    pass
        except Exception:
            _tts = TTS(MODEL_NAME)
    return _tts


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/tts")
async def tts(body: Dict[str, Any]):
    text = body.get("text") or ""
    speaker = body.get("voice")
    if not text:
        return JSONResponse(status_code=400, content={"error": "missing text"})
    tts = get_tts()
    wav = tts.tts(text=text, speaker=speaker)  # type: ignore
    wav = np.array(wav, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, 22050, format="WAV")
    buf.seek(0)
    # return base64 for simplicity
    import base64
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"audio_wav_base64": b64, "sample_rate": 22050}


