from __future__ import annotations

from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="MelodyWriter", version="0.1.0")


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


