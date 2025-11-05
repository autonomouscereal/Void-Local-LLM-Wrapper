from __future__ import annotations
# WARNING: Do NOT add SQLAlchemy to this service. Use asyncpg with proper pooling and raw SQL only.
#
# Historical context (high-signal notes, not exhaustive):
# - Original implementation sometimes used SQLAlchemy/psycopg; this project now forbids it.
#   We use asyncpg exclusively with connection pooling and raw SQL. All JSONB writes go through
#   json.dumps(... )::jsonb to avoid type errors (e.g., "expected str, got dict").
# - The browser previously showed NetworkError/uncaught promise rejections while the orchestrator
#   actually completed later with 200 OK. Root causes we mitigated here:
#     * Potential chunked transfer/keep-alive quirks: We now fully materialize bodies, set
#       explicit Content-Length, and send Connection: close on proxy responses to force the browser
#       to treat the response as complete.
#     * CORS inconsistencies: We add global CORS headers to every response and short-circuit OPTIONS
#       preflights on any path.
#     * Duplicated frontend POSTs: The UI now sends exactly one awaited POST to the chat endpoint
#       and waits for completion; no polling or background fallbacks.
#     * Hidden timeouts/proxies: httpx is used with timeout=None and trust_env=False so no implicit
#       time limits or environment proxies can abort the upstream request.
# - For JSON robustness, we rely on a custom JSONParser when decoding inputs or upstream responses.
#   We also re-serialize JSON on output to normalize payloads before returning to the browser.

import os
import json
from typing import Any, Dict, List, Optional
import time
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
import asyncpg
from fastapi.middleware.cors import CORSMiddleware
from .json_parser import JSONParser
import logging
import websockets  # type: ignore

# Create app BEFORE any decorators use it
app = FastAPI(title="Chat UI Backend", version="0.1.0")
# NOTE: We rely on the custom global CORS middleware below for deterministic headers and
# OPTIONS short-circuiting. Built-in CORSMiddleware is intentionally not used to avoid
# double-handling and potential interference.


ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")


async def init_pool() -> asyncpg.pool.Pool:
    # Asyncpg pool with auto-DDL for required tables. No ORM anywhere.
    pool = await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
        min_size=1,
        max_size=10,
    )
    async with pool.acquire() as conn:
        # conversations/messages/attachments are created up-front so cold-starts succeed
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
              id BIGSERIAL PRIMARY KEY,
              title TEXT,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
              id BIGSERIAL PRIMARY KEY,
              conversation_id BIGINT REFERENCES conversations(id) ON DELETE CASCADE,
              role TEXT,
              content JSONB,
              created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attachments (
              id BIGSERIAL PRIMARY KEY,
              conversation_id BIGINT REFERENCES conversations(id) ON DELETE CASCADE,
              name TEXT,
              url TEXT,
              mime TEXT,
              created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
    return pool


pool: asyncpg.pool.Pool | None = None

@app.on_event("startup")
async def _startup():
    global pool
    pool = await init_pool()

def _pool() -> asyncpg.pool.Pool:
    assert pool is not None, "DB pool not initialized"
    return pool
def _decode_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        # Use custom JSON parser only when the string looks like JSON; otherwise
        # treat it as plain text to avoid over-eager coercion.
        txt = value.strip()
        if txt.startswith("{") or txt.startswith("["):
            try:
                parser = JSONParser()
                expected = {"text": str}
                obj = parser.parse(txt, expected)
                # If the normalized parse still doesn't look like a dict, fall back to raw text
                return obj if isinstance(obj, dict) and obj else {"text": value}
            except Exception:
                return {"text": value}
        return {"text": value}
    return {"text": str(value or "")}

# Mount static AFTER API routes to avoid intercepting /api/* with 404/405


logging.basicConfig(level=logging.INFO)
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    # Last-resort guard: always send a JSON body with explicit length so the browser receives bytes
    logging.exception("UNHANDLED EXCEPTION processing %s %s", request.method, request.url.path)
    payload = {"error": "proxy failure", "detail": str(exc)}
    body = json.dumps(payload)
    headers = {
        "Cache-Control": "no-store",
        "Connection": "close",
        "Content-Length": str(len(body.encode("utf-8"))),
        "Content-Type": "application/json; charset=utf-8",
    }
    return Response(content=body, status_code=500, media_type="application/json", headers=headers)


# Runtime notes (historic attempts and why we chose the current design):
#
# HIGH-LEVEL ISSUE
# - Symptom: Browser reports NetworkError/uncaught promise and shows 0 bytes received; meanwhile orchestrator later
#   completes with 200. This strongly implies the request fails at UI/proxy layer (preflight/headers/connection), not
#   at the orchestrator.
#
# WHAT WE TRIED (CHRONOLOGICAL SUMMARY)
# 1) CORS everywhere
#    - Added global CORS middleware stamping headers on all responses; OPTIONS short-circuit for any path.
#    - Added explicit OPTIONS endpoints for /api/chat and /api/conversations/{cid}/chat.
#    - Tried both narrow header sets (Content-Type/Accept) and permissive '*' for methods/headers/expose.
#    - Current: permissive '*' in global middleware; per-endpoint CORS headers removed to avoid conflicts.
#
# 2) Response framing quirks
#    - Tried plain JSONResponse; tried raw Response with Content-Length + Connection: close to avoid chunked transfer.
#    - Removed manual Connection/Content-Length mutations in early versions, then reintroduced explicit lengths to
#      eliminate browser ambiguity. Current: explicit length + close for non-stream responses.
#
# 3) Fetch vs XHR in the UI
#    - Swapped between fetch() and XMLHttpRequest to bypass fetch-specific abort behaviors.
#    - Added detailed client-side timing/logs; still observed 0 bytes in some cases.
#    - Current: UI supports a single awaited POST with robust error surfacing; XHR variant is in place.
#
# 4) Duplicate POSTs/polling
#    - Removed duplicate POST flows and polling fallbacks that previously caused confusing timing artifacts.
#    - Current: exactly one POST per send action.
#
# 5) Streaming/SSE
#    - Tried streaming keepalives (whitespace) to keep the connection from idling out mid-flight.
#    - Result: did not change the observed failure; reverted to simple awaited POST.
#
# 6) SQLAlchemy removal / asyncpg
#    - Fully removed SQLAlchemy/psycopg adapters; replaced with asyncpg pool + raw SQL.
#    - Auto-DDL for conversations/messages/attachments on startup.
#    - Ensured JSONB writes use json.dumps(... )::jsonb consistently (fixed 'expected str, got dict').
#
# 7) Custom JSON parser correctness & safety
#    - Ensured parser.parse is used as the single entrypoint; added non-destructive "pristine" parse before repairs.
#    - Fixed TypeErrors in ensure_structure/select_best_result for list/dict schemas.
#    - Wrapped request-body parse in guards to never abort the request path; fallback to {"content": decoded}.
#    - Removed file logging in parser to avoid potential I/O stalls.
#
# 8) Path isolation
#    - Added /api/chat passthrough-style endpoint to avoid any path-specific interceptors that might affect /chat.
#    - Temporarily pointed UI to /api/chat to test path neutrality; no change reported.
#
# WHAT STILL ISN'T WORKING
# - In the user's environment, the browser sees 0 bytes and no status label for the chat POST in DevTools. Backend logs
#   show the proxy receives the request, logs the upstream call, and later receives 200 from orchestrator.
# - This suggests the response cannot leave the proxy (connection killed or blackholed) before headers are written,
#   or the client stops listening immediately upon send.
#
# CURRENT DESIGN CHOICES (to minimize app-level causes)
# - Global CORS middleware only; no per-endpoint CORS mutations to avoid duplication/conflicts.
# - Responses use explicit Content-Length and Connection: close for deterministic framing.
# - HTTP client uses timeout=None and trust_env=False to ignore env proxies/timeouts.
# - No streaming; single awaited POST end-to-end; assistant persistence is best-effort and cannot block the response.
# - Parser failures cannot abort the request; request parsing degrades to raw text; response relay is byte-for-byte.
#
# NEXT TRIAGE HINTS (outside of code changes)
# - If POST still shows 0 bytes and no status label, verify no extensions/filters are intercepting requests,
#   and confirm container ingress/reverse proxy is not dropping connections on specific paths.


@app.middleware("http")
async def global_cors_middleware(request: Request, call_next):
    logging.info("REQ %s %s", request.method, request.url.path)
    # Preflight short-circuit for any path: respond quickly and add permissive headers
    if request.method == "OPTIONS":
        return Response(status_code=204, headers={
            "Access-Control-Allow-Origin": (request.headers.get("origin") or "*"),
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "false",
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Max-Age": "86400",
            "Access-Control-Allow-Private-Network": "true",
            "Connection": "close",
            "Vary": "Origin",
        })
    resp = await call_next(request)
    logging.info("RES %s %s %s", request.method, request.url.path, getattr(resp, 'status_code', ''))
    # Inject permissive CORS headers on every response
    resp.headers["Access-Control-Allow-Origin"] = request.headers.get("origin") or "*"
    resp.headers["Access-Control-Allow-Methods"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Credentials"] = "false"
    resp.headers["Access-Control-Expose-Headers"] = "*"
    resp.headers["Access-Control-Max-Age"] = "86400"
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    resp.headers["Vary"] = "Origin"
    # Help browsers load media without ORB/CORP issues when proxied through this backend
    if request.url.path.startswith("/uploads/") or request.url.path.endswith(".mp4") or request.url.path.endswith(".png"):
        resp.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
        resp.headers.setdefault("Timing-Allow-Origin", "*")
    return resp


@app.post("/api/conversations")
async def create_conversation(body: Dict[str, Any]):
    title = (body or {}).get("title") or "New Conversation"
    async with _pool().acquire() as conn:
        row = await conn.fetchrow("INSERT INTO conversations (title) VALUES ($1) RETURNING id", title)
    return {"id": row[0], "title": title}


@app.get("/api/conversations")
async def list_conversations():
    async with _pool().acquire() as conn:
        rows = await conn.fetch("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 100")
        data = [{"id": r["id"], "title": r["title"], "created_at": str(r["created_at"]), "updated_at": str(r["updated_at"]) } for r in rows]
    return {"data": data}


@app.get("/api/conversations/{cid}/messages")
async def list_messages(cid: int):
    async with _pool().acquire() as conn:
        rows = await conn.fetch("SELECT id, role, content, created_at FROM messages WHERE conversation_id=$1 ORDER BY id ASC", cid)
        return {"data": [{"id": r["id"], "role": r["role"], "content": _decode_json(r["content"]), "created_at": str(r["created_at"]) } for r in rows]}


@app.post("/api/upload")
async def upload(conversation_id: int = Form(...), file: UploadFile = File(...)):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; use orchestrator /upload directly"})


@app.post("/api/attachments.add")
async def attachments_add(body: Dict[str, Any]):
    cid = int((body or {}).get("conversation_id") or 0)
    name = (body or {}).get("name") or ""
    url = (body or {}).get("url") or ""
    mime = (body or {}).get("mime") or "application/octet-stream"
    if not cid or not url:
        return JSONResponse(status_code=400, content={"error": "missing conversation_id or url"})
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO attachments (conversation_id, name, url, mime) VALUES ($1, $2, $3, $4)", cid, name, url, mime)
    return {"ok": True}


@app.get("/api/conversations/{cid}/attachments")
async def list_attachments(cid: int):
    async with _pool().acquire() as conn:
        rows = await conn.fetch("SELECT id, name, url, mime FROM attachments WHERE conversation_id=$1 ORDER BY id", cid)
        return {"data": [{"id": r["id"], "name": r["name"], "url": r["url"], "mime": r["mime"] } for r in rows]}


@app.post("/api/conversations/{cid}/message")
async def add_message(cid: int, body: Dict[str, Any]):
    role = (body or {}).get("role") or "user"
    content = (body or {}).get("content") or ""
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3::jsonb)", cid, role, json.dumps({"text": content}))
    return {"ok": True}


def _build_openai_messages(base: List[Dict[str, Any]], attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Reuse attachments implicitly by adding a system hint listing URLs; messages can still pass content arrays
    if not attachments:
        return base
    sys = {"role": "system", "content": "Conversation attachments available:\n" + json.dumps(attachments)}
    return [sys] + base


@app.post("/api/conversations/{cid}/chat")
async def chat(cid: int, request: Request, background_tasks: BackgroundTasks):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})


@app.options("/api/conversations/{cid}/chat")
async def chat_preflight(cid: int):
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Max-Age": "86400",
    })


@app.options("/api/{path:path}")
async def any_preflight(path: str):
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Max-Age": "86400",
    })


# Alternate chat endpoint to avoid any extension/content filters on "/chat" path
@app.post("/api/chat")
async def chat_alt(body: Dict[str, Any], background_tasks: BackgroundTasks):
    logging.info("/api/chat: start cid=%s", (body or {}).get("conversation_id"))
    cid = int((body or {}).get("conversation_id") or 0)
    user_content = (body or {}).get("content") or ""
    if not cid:
        return JSONResponse(status_code=400, content={"error": "missing conversation_id"})
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2::jsonb)", cid, json.dumps({"text": user_content}))
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)
    oa_msgs = _build_openai_messages([{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False, "cid": cid}
    async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
        url = ORCH_URL.rstrip("/") + "/v1/chat/completions"
        logging.info("proxy -> orchestrator POST %s", url)
        rr = await client.post(url, json=payload)
    # best-effort assistant extraction (does not affect response)
    if (rr.headers.get("content-type") or "").startswith("application/json"):
        try:
            expected_response = {"choices": [{"message": {"content": str}}]}
            obj = JSONParser().parse(rr.text, expected_response)
            assistant = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content")
            if assistant:
                async with _pool().acquire() as c2:
                    await c2.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2::jsonb)", cid, json.dumps({"text": assistant}))
        except Exception:
            pass
    body = rr.content
    ct = rr.headers.get("content-type") or "application/json; charset=utf-8"
    headers = {
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Expose-Headers": "Content-Type, Content-Length",
        "Connection": "close",
        "Content-Length": str(len(body)),
        "Content-Type": ct,
    }
    logging.info("/api/chat: done cid=%s", cid)
    return Response(content=body, media_type=ct, status_code=rr.status_code, headers=headers)


# Neutral path versions (avoid filters on "chat")
@app.post("/api/conversations/{cid}/call")
async def call_conv(cid: int, request: Request, background_tasks: BackgroundTasks):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})


@app.post("/api/call")
async def call_alt(body: Dict[str, Any], background_tasks: BackgroundTasks):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})


@app.post("/api/echo")
async def echo(body: Dict[str, Any]):
    logging.info("/api/echo: start")
    data = {"ok": True, "body": body}
    b = json.dumps(data)
    headers = {
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Connection": "close",
        "Content-Length": str(len(b.encode("utf-8"))),
    }
    logging.info("/api/echo: done")
    return Response(content=b, media_type="application/json", headers=headers)


@app.post("/api/passthrough")
async def passthrough(body: Dict[str, Any]):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})


@app.get("/api/tool.stream")
async def tool_stream(name: str, request: Request):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})

@app.options("/api/chat")
async def chat_alt_preflight():
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Max-Age": "86400",
    })


@app.get("/api/orchestrator/diagnostics")
async def orch_diag():
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/api/conversations/{cid}/chat")
async def chat_get(cid: int, content: str = ""):
    # Convenience GET for environments where POST fetch is blocked; DO NOT use in production
    user_content = content or ""
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2::jsonb)", cid, json.dumps({"text": user_content}))
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)
    oa_msgs = _build_openai_messages([{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}
    async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
        rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
        expected_response = {
            "choices": [
                {"message": {"content": str}}
            ]
        }
        data = JSONParser().parse(rr.text, expected_response)
    assistant = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2::jsonb)", cid, json.dumps({"text": assistant}))
    body = json.dumps(data)
    headers = {
        "Cache-Control": "no-store",
        "Connection": "close",
        "Content-Length": str(len(body.encode("utf-8"))),
        "Access-Control-Allow-Origin": "*",
    }
    return Response(content=body, media_type="application/json", headers=headers)


@app.get("/api/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})


@app.websocket("/api/tool.ws")
async def tool_ws_proxy(websocket: WebSocket):
    await websocket.close(code=1008)

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_proxy(job_id: str, interval_ms: Optional[int] = None):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})

@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_proxy(job_id: str):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; call orchestrator directly"})


# Lightweight proxies for orchestrator admin/capabilities so the frontend can hit same-origin
@app.get("/capabilities.json")
async def capabilities_proxy():
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
            r = await client.get(ORCH_URL.rstrip("/") + "/capabilities.json")
        body = r.content
        headers = {
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Private-Network": "true",
            "Access-Control-Expose-Headers": "Content-Type, Content-Length",
            "Connection": "close",
            "Content-Length": str(len(body)),
            "Content-Type": r.headers.get("content-type") or "application/json",
        }
        return Response(content=body, media_type=headers["Content-Type"], status_code=r.status_code, headers=headers)
    except Exception as ex:
        return JSONResponse(status_code=502, content={"error": str(ex)})


@app.get("/jobs.list")
async def jobs_list_proxy():
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
            r = await client.get(ORCH_URL.rstrip("/") + "/jobs.list")
        body = r.content
        headers = {
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Private-Network": "true",
            "Access-Control-Expose-Headers": "Content-Type, Content-Length",
            "Connection": "close",
            "Content-Length": str(len(body)),
            "Content-Type": r.headers.get("content-type") or "application/json",
        }
        return Response(content=body, media_type=headers["Content-Type"], status_code=r.status_code, headers=headers)
    except Exception as ex:
        return JSONResponse(status_code=502, content={"error": str(ex)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/uploads/{path:path}")
async def uploads_proxy(path: str, request: Request):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; use orchestrator /uploads directly"})


@app.websocket("/api/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.close(code=1008)


app.mount("/", StaticFiles(directory="/app/frontend_dist", html=True), name="static")
# Static assets are mounted at root, AFTER API routes are declared to avoid intercepting /api/*. 


