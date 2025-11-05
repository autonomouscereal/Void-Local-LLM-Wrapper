from __future__ import annotations

from typing import Dict, Any
from fastapi import FastAPI

app = FastAPI(title="Mixer (stub)", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/mix")
async def mix(body: Dict[str, Any]):
    # Stub echo
    return {"uri": "", "lufs": -14.0}


