from __future__ import annotations

import logging, sys
from fastapi import FastAPI
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Any, Dict

# Absolute imports to avoid relative/cycle resolution issues under Uvicorn
from .routes import run_all as run_all_routes
from .routes import logs as logs_routes
from .routes import tools as tools_routes
from .routes import toolrun as toolrun_routes
from app.middleware.ws_permissive import PermissiveWebSocketMiddleware
from app.middleware.pna import AllowPrivateNetworkMiddleware
from app.middleware.cors_extra import AppendCommonHeadersMiddleware


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
# Short-circuit all preflight OPTIONS with explicit 200 + zero-length body to avoid 204/chunked quirks
# Remove legacy Preflight204Middleware in favor of explicit middleware below

# Append Access-Control-Allow-Private-Network on preflight (Chromium PNA)
app.add_middleware(AllowPrivateNetworkMiddleware)

# Global CORS (wide open, reflect Origin on non-OPTIONS)
@app.middleware("http")
async def _cors_open(request: Request, call_next):
    # Preflight: explicit 200 with deterministic headers
    if request.method.upper() == "OPTIONS":
        origin = request.headers.get("origin") or request.headers.get("Origin") or "*"
        acrh = request.headers.get("access-control-request-headers") or "*"
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": acrh,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Allow-Private-Network": "true",
            "Connection": "close",
            "Vary": "Origin",
            "Content-Length": "0",
        }
        return Response(status_code=200, headers=headers)
    resp = await call_next(request)
    origin = request.headers.get("origin") or request.headers.get("Origin") or "*"
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Credentials"] = "true"
    resp.headers["Access-Control-Expose-Headers"] = "*"
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    resp.headers["Vary"] = "Origin"
    return resp

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
from app.main import v1_image_generate as _v1_image_generate  # type: ignore
from app.main import post_image_dispatch as _post_image_dispatch  # type: ignore
from app.main import upload as _upload  # type: ignore
from app.main import healthz as _healthz  # type: ignore
from app.main import capabilities as _capabilities  # type: ignore
from app.main import (
    admin_jobs_list as _admin_jobs_list,
    admin_jobs_replay as _admin_jobs_replay,
    jobs_start as _jobs_start,
    create_job as _create_job,
    get_job as _get_job,
    list_jobs as _list_jobs,
    stream_job as _stream_job,
    cancel_job as _cancel_job,
)  # type: ignore
_v1 = APIRouter(prefix="/v1")
@_v1.post("/chat/completions")
async def _v1_chat_completions(body: Dict[str, Any], request: Request):
	return await _chat_completions(body, request)

@_v1.post("/image.generate")
async def _v1_image_generate(request: Request):
	return await _v1_image_generate(request)

@_v1.post("/image/dispatch")
async def _v1_image_dispatch(request: Request):
	return await _post_image_dispatch(request)
app.include_router(_v1)

# Uploads and health/admin
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
	return await _upload(file)

@app.get("/healthz")
async def healthz():
	return await _healthz()

@app.get("/capabilities.json")
async def capabilities():
	return await _capabilities()

# Jobs API parity
@app.post("/jobs/start")
async def jobs_start(body: Dict[str, Any]):
	return await _jobs_start(body)

@app.get("/jobs.list")
async def jobs_list_admin():
	return await _admin_jobs_list()

@app.get("/jobs.replay")
async def jobs_replay_admin(cid: str):
	return await _admin_jobs_replay(cid)

@app.post("/jobs")
async def jobs_create(body: Dict[str, Any]):
	return await _create_job(body)

@app.get("/jobs")
async def jobs_list(status: str | None = None, limit: int = 50, offset: int = 0):
	return await _list_jobs(status, limit, offset)

@app.get("/jobs/{job_id}")
async def jobs_get(job_id: str):
	return await _get_job(job_id)

@app.get("/jobs/{job_id}/stream")
async def jobs_stream(job_id: str, interval_ms: int | None = None):
	return await _stream_job(job_id, interval_ms)

@app.post("/jobs/{job_id}/cancel")
async def jobs_cancel(job_id: str):
	return await _cancel_job(job_id)

