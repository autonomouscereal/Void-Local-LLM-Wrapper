from __future__ import annotations

from typing import Any, Dict
from fastapi import FastAPI

app = FastAPI(title="Prosody Predictor (stub)", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/suggest")
async def suggest(body: Dict[str, Any]):
    score = body.get("score_json") or {}
    notes = score.get("notes") or []
    # naive: stretch long vowels by +5%, tiny pitch nudge +10 cents on every 4th note
    stretch = []
    pitch = []
    for i, _ in enumerate(notes):
        if i % 4 == 0:
            pitch.append({"note_idx": i, "cents": 10})
        if i % 3 == 0:
            stretch.append({"note_idx": i, "ratio": 1.05})
    return {"note_stretch": stretch[:16], "pitch_nudge_cents": pitch[:16]}


