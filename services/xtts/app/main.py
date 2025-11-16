from __future__ import annotations

import io
import os
from typing import Dict, Any

import soundfile as sf
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from TTS.api import TTS  # type: ignore
import torch


MODEL_NAME = os.getenv("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")

app = FastAPI(title="XTTS TTS Service", version="0.1.0")

_tts = None


def get_tts():
    global _tts
    if _tts is None:
        use_cuda = torch.cuda.is_available()
        _tts = TTS(MODEL_NAME)  # may download/initialize; failures should surface loudly
        if use_cuda:
            # Some TTS models expose .to(); ignore if unsupported
            _tts.to("cuda")  # type: ignore[arg-type]
    return _tts


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/tts")
async def tts(body: Dict[str, Any]):
    trace_id = body.get("trace_id") if isinstance(body.get("trace_id"), str) else "tt_xtts_unknown"
    text = body.get("text") or ""
    speaker = body.get("voice")
    language = body.get("language") or "en"
    if not text:
        return JSONResponse(
            status_code=400,
            content={
                "schema_version": 1,
                "trace_id": trace_id,
                "ok": False,
                "result": None,
                "error": {"code": "bad_request", "message": "Missing 'text' for TTS."},
            },
        )
    tts = get_tts()
    wav = tts.tts(text=text, speaker=speaker, language=language)  # type: ignore[call-arg]
    wav = np.array(wav, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, 22050, format="WAV")
    buf.seek(0)
    import base64
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return JSONResponse(
        status_code=200,
        content={
            "schema_version": 1,
            "trace_id": trace_id,
            "ok": True,
            "result": {"audio_wav_base64": b64, "sample_rate": 22050, "language": language},
            "error": None,
        },
    )


