from __future__ import annotations

from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="RVC (stub)", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/convert")
async def convert(body: Dict[str, Any]):
    if not body.get("voice_lock_id"):
        return JSONResponse(status_code=400, content={"error": "missing voice_lock_id"})
    # Echo behavior for stub
    return {"audio_wav_base64": ""}


