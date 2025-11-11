from __future__ import annotations

from fastapi import FastAPI

# Absolute imports to avoid relative/cycle resolution issues under Uvicorn
from app.routes import run_all as run_all_routes


app = FastAPI(title="Void Orchestrator")

# Include routers once; no startup handlers or import-time side effects here
app.include_router(run_all_routes.router)


@app.get("/_alive")
def _alive():
    return {"ok": True}


