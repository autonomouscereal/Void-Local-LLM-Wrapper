from __future__ import annotations

import logging, sys
from fastapi import FastAPI
from fastapi import APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

# Absolute imports to avoid relative/cycle resolution issues under Uvicorn
from .routes import run_all as run_all_routes
from .routes import logs as logs_routes
from .routes import tools as tools_routes
from .routes import toolrun as toolrun_routes
from app.middleware.ws_permissive import PermissiveWebSocketMiddleware
from app.middleware.pna import AllowPrivateNetworkMiddleware
from app.middleware.cors_extra import AppendCommonHeadersMiddleware
from app.middleware.preflight import Preflight204Middleware


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	stream=sys.stdout,
)

app = FastAPI(title="Void Orchestrator")

# Include routers once; no startup handlers or import-time side effects here
app.include_router(run_all_routes.router)
app.include_router(logs_routes.router)
app.include_router(tools_routes.router)
app.include_router(toolrun_routes.router)

# IMPORTANT (policy): Do not change the container workpath/module path.
# All imports assume the 'app.*' module namespace rooted at /app.
# Short-circuit all preflight OPTIONS with 204 + full headers — must be outermost
app.add_middleware(Preflight204Middleware)

# Append Access-Control-Allow-Private-Network on preflight (Chromium PNA)
app.add_middleware(AllowPrivateNetworkMiddleware)

# Global CORS (open by default per project rules)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",  # reflect caller Origin (works with credentials)
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Append common headers on all responses (including static and non-API)
app.add_middleware(AppendCommonHeadersMiddleware)

# WebSocket middleware to strip Origin header
app.add_middleware(PermissiveWebSocketMiddleware)

@app.get("/_alive")
def _alive():
    return {"ok": True}

# Include OpenAI-compatible chat completions endpoint from the canonical implementation,
# mounted via a local router to guarantee the exact path and method.
from app.main import chat_completions as _chat_completions  # type: ignore
_v1 = APIRouter(prefix="/v1")
@_v1.post("/chat/completions")
async def _v1_chat_completions(body: Dict[str, Any], request: Request):
	return await _chat_completions(body, request)
app.include_router(_v1)


