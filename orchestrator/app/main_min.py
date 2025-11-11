from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Absolute imports to avoid relative/cycle resolution issues under Uvicorn
from app.routes import run_all as run_all_routes
from app.middleware.ws_permissive import PermissiveWebSocketMiddleware
from app.middleware.pna import AllowPrivateNetworkMiddleware


app = FastAPI(title="Void Orchestrator")

# Include routers once; no startup handlers or import-time side effects here
app.include_router(run_all_routes.router)

# Append Access-Control-Allow-Private-Network on preflight (Chromium PNA) — must be outermost
app.add_middleware(AllowPrivateNetworkMiddleware)

# Global CORS (open by default per project rules)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",  # reflect caller Origin (works with credentials)
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# WebSocket middleware to strip Origin header
app.add_middleware(PermissiveWebSocketMiddleware)

@app.get("/_alive")
def _alive():
    return {"ok": True}


