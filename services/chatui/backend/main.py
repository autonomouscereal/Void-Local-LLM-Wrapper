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
from urllib.parse import parse_qs

import httpx
from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
import asyncpg
from fastapi.middleware.cors import CORSMiddleware
from .json_parser import JSONParser
import logging

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
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "false",
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Max-Age": "86400",
            "Access-Control-Allow-Private-Network": "true",
            "Connection": "close",
        })
    resp = await call_next(request)
    logging.info("RES %s %s %s", request.method, request.url.path, getattr(resp, 'status_code', ''))
    # Inject permissive CORS headers on every response
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Credentials"] = "false"
    resp.headers["Access-Control-Expose-Headers"] = "*"
    resp.headers["Access-Control-Max-Age"] = "86400"
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
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
    # Forward to orchestrator /upload to get a URL, then store attachment
    try:
        # Read file and send to orchestrator's upload endpoint
        content = await file.read()
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            files = {"file": (file.filename, content, file.content_type)}
            r = await client.post(ORCH_URL.rstrip("/") + "/upload", files=files)
            r.raise_for_status()
            # Avoid native json(); return raw upstream text
            data = r.text
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO attachments (conversation_id, name, url, mime) VALUES ($1, $2, $3, $4)", conversation_id, file.filename, data.get("url"), file.content_type)
    return {"ok": True, "url": data.get("url"), "name": data.get("name")}


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
    t0 = time.perf_counter()
    logging.info("/api/conversations/%s/chat: start", cid)
    # Request body parsing:
    # - Accept JSON, raw text, or x-www-form-urlencoded
    # - Prefer framework JSON parsing when available; fallback to custom parser
    ct = (request.headers.get("content-type") or "").lower()
    body: Dict[str, Any] = {}
    parser = JSONParser()
    # Expected request shape for normalization – parser will ensure defaults
    expected_request = {
        "conversation_id": int,
        "content": str,
        "messages": [
            {"role": str, "content": str}
        ],
    }
    if "application/json" in ct:
        raw_bytes = await request.body()
        decoded = raw_bytes.decode("utf-8", errors="replace")
        try:
            body = parser.parse(decoded, expected_request)
        except Exception:
            # Never let parsing kill the request path; degrade gracefully
            body = {"content": decoded}
    else:
        raw_bytes = await request.body()
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        if "application/x-www-form-urlencoded" in ct:
            form = parse_qs(raw_text, keep_blank_values=True)
            if "content" in form and len(form["content"]) > 0:
                candidate = form["content"][0]
                # Attempt full parse/normalize; if not JSON, fall back to treating as raw content
                try:
                    body = parser.parse(candidate, expected_request)
                except Exception:
                    body = {"content": candidate}
            elif len(form) == 1:
                first_key = next(iter(form))
                vals = form.get(first_key) or [raw_text]
                candidate = vals[0]
                try:
                    body = parser.parse(candidate, expected_request)
                except Exception:
                    body = {"content": candidate}
            else:
                body = {"content": raw_text}
        else:
            # Treat as raw text
            body = {"content": raw_text}
    logging.info("chat request cid=%s ct=%s keys=%s", cid, ct, list(body.keys()))
    user_content = (body or {}).get("content") or ""
    messages = (body or {}).get("messages") or []
    # Streaming is disabled here; we proxy non-stream to simplify the browser path
    stream = False
    # Store user message in JSONB; ALWAYS json.dumps(... )::jsonb to match DB types
    t1 = time.perf_counter()
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2::jsonb)", cid, json.dumps({"text": user_content}))
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)
    logging.info("db:user_insert+attachments cid=%s ms=%.2f", cid, (time.perf_counter() - t1) * 1000)

    # Build messages for orchestrator
    oa_msgs = _build_openai_messages(messages + [{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}

    async def _proxy_stream():
        try:
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                url = ORCH_URL.rstrip("/") + "/v1/chat/completions"
                logging.info("proxy -> orchestrator POST %s", url)
                async with client.stream("POST", url, json=payload) as r:
                    ct = r.headers.get("content-type", "")
                    if stream and ct.startswith("text/event-stream"):
                        async for chunk in r.aiter_text():
                            # pass through SSE chunks unchanged
                            yield chunk
                    else:
                        # Non-SSE: forward raw bytes exactly to avoid browser/client quirks
                        body = await r.aread()
                        yield body
        except Exception as ex:
            yield json.dumps({"error": str(ex)})

    # Keepalive streaming: periodically yield whitespace while waiting for upstream, then yield body
    t2 = time.perf_counter()
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        url = ORCH_URL.rstrip("/") + "/v1/chat/completions"
        logging.info("proxy -> orchestrator POST %s", url)
        rr = await client.post(url, json=payload)
    t3 = time.perf_counter()
    ct = rr.headers.get("content-type") or "application/octet-stream"
    status = rr.status_code
    if ct.startswith("application/json"):
        try:
            expected_response = {"choices": [{"message": {"content": str}}]}
            obj = JSONParser().parse(rr.text, expected_response)
            assistant_text = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content")
            if assistant_text:
                async with _pool().acquire() as c2:
                    await c2.execute(
                        "INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2::jsonb)",
                        cid,
                        json.dumps({"text": assistant_text}),
                    )
        except Exception:
            pass
        body_bytes = rr.content
        headers = {
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Private-Network": "true",
            "Access-Control-Expose-Headers": "Content-Type, Content-Length",
            "Connection": "close",
            "Content-Length": str(len(body_bytes)),
            "Content-Type": ct if ct else "application/json; charset=utf-8",
        }
        logging.info("/api/conversations/%s/chat: done status=%s ct=%s t_orch_ms=%.2f t_total_ms=%.2f", cid, status, ct, (t3 - t2) * 1000, (time.perf_counter() - t0) * 1000)
        return Response(content=body_bytes, media_type="application/json", status_code=status, headers=headers)
    raw = rr.content
    headers = {
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Expose-Headers": "Content-Type, Content-Length",
        "Connection": "close",
        "Content-Length": str(len(raw)),
        "Content-Type": ct,
    }
    logging.info("/api/conversations/%s/chat: done status=%s ct=%s t_orch_ms=%.2f t_total_ms=%.2f", cid, status, ct, (t3 - t2) * 1000, (time.perf_counter() - t0) * 1000)
    return Response(content=raw, media_type=ct, status_code=status, headers=headers)


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
    payload = {"messages": oa_msgs, "stream": False}
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
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
    logging.info("/api/conversations/%s/call: start", cid)
    parser = JSONParser()
    try:
        # Per policy use parser, not request.json()
        raw_bytes = await request.body()
        decoded = raw_bytes.decode("utf-8", errors="replace")
        expected_request = {"content": str}
        payload_in = parser.parse(decoded, expected_request)
    except Exception:
        raw_bytes = await request.body()
        decoded = raw_bytes.decode("utf-8", errors="replace")
        try:
            payload_in = parser.method_json_loads(decoded)
        except Exception:
            repaired = parser.attempt_repair(decoded)
            payload_in = parser.method_json_loads(repaired)
    user_content = (payload_in or {}).get("content") or ""
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2::jsonb)", cid, json.dumps({"text": user_content}))
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)
    oa_msgs = _build_openai_messages([{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        url = ORCH_URL.rstrip("/") + "/v1/chat/completions"
        logging.info("proxy -> orchestrator POST %s", url)
        rr = await client.post(url, json=payload)
        try:
            expected_response = {
                "choices": [
                    {"message": {"content": str}}
                ]
            }
            data = JSONParser().parse(rr.text, expected_response)
        except Exception:
            data = {"choices": [{"message": {"content": ""}}]}
    assistant = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    async def _persist_assistant_message(conv_id: int, text_value: str) -> None:
        try:
            async with _pool().acquire() as c2:
                await c2.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2::jsonb)", conv_id, json.dumps({"text": text_value}))
        except Exception:
            logging.exception("persist assistant failed cid=%s", conv_id)
    background_tasks.add_task(_persist_assistant_message, cid, assistant)
    body = json.dumps(data)
    headers = {
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Expose-Headers": "*",
        "Connection": "close",
        "Content-Length": str(len(body.encode("utf-8"))),
    }
    logging.info("/api/conversations/%s/call: done", cid)
    return Response(content=body, media_type="application/json", headers=headers)


@app.post("/api/call")
async def call_alt(body: Dict[str, Any], background_tasks: BackgroundTasks):
    cid = int((body or {}).get("conversation_id") or 0)
    logging.info("/api/call: start cid=%s", cid)
    user_content = (body or {}).get("content") or ""
    if not cid:
        return JSONResponse(status_code=400, content={"error": "missing conversation_id"})
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2::jsonb)", cid, json.dumps({"text": user_content}))
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)
    oa_msgs = _build_openai_messages([{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
    # best-effort assistant persistence without affecting response
    def _persist_from_response(conv_id: int, status_code: int, content_type: str, text_body: str) -> None:
        if content_type.startswith("application/json") and status_code < 500 and (text_body.strip().startswith('{') or text_body.strip().startswith('[')):
            parser = JSONParser()
            expected_response = {
                "choices": [
                    {"message": {"content": str}}
                ]
            }
            obj = parser.parse(text_body, expected_response)
            assistant_text = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content")
            if assistant_text:
                import asyncio
                async def _save() -> None:
                    async with _pool().acquire() as c2:
                        await c2.execute(
                            "INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2::jsonb)",
                            conv_id,
                            json.dumps({"text": assistant_text}),
                        )
                asyncio.get_event_loop().create_task(_save())
    ct = rr.headers.get("content-type") or "application/json"
    logging.info("/api/call: upstream status=%s ct=%s", rr.status_code, ct)
    background_tasks.add_task(_persist_from_response, cid, rr.status_code, ct, rr.text)
    # Relay response; for JSON use framework JSONResponse (lets Starlette set headers)
    # Always return explicit length and close connection
    body = rr.content
    headers = {
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Expose-Headers": "Content-Type, Content-Length",
        "Connection": "close",
        "Content-Length": str(len(body)),
        "Content-Type": ct,
    }
    logging.info("/api/call: done cid=%s", cid)
    return Response(content=body, media_type=ct, status_code=rr.status_code, headers=headers)


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
    # Minimal proxy: no DB, no mutation; just relay to orchestrator and return raw
    logging.info("/api/passthrough: start")
    user_content = (body or {}).get("content") or ""
    payload = {"messages": [{"role": "user", "content": user_content}], "stream": False}
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
    except Exception as ex:
        logging.exception("/api/passthrough proxy error")
        return JSONResponse(status_code=502, content={"error": str(ex)})
    ct = rr.headers.get("content-type") or "application/json"
    headers = {
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Expose-Headers": "*",
        "Connection": "close",
    }
    logging.info("/api/passthrough: done status=%s", rr.status_code)
    return Response(content=rr.content, media_type=ct, status_code=rr.status_code, headers=headers)


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
    out: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            h = await client.get(ORCH_URL.rstrip("/") + "/healthz")
            out["healthz_status"] = h.status_code
            # Diagnostics: return raw text to avoid parser coupling
            out["healthz_body"] = h.text
    except Exception as ex:
        out["healthz_error"] = str(ex)
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            d = await client.get(ORCH_URL.rstrip("/") + "/debug")
            out["debug_status"] = d.status_code
            out["debug_body"] = d.text
    except Exception as ex:
        out["debug_error"] = str(ex)
    return out


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
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
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
    params = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.get(ORCH_URL.rstrip("/") + "/jobs", params=params)
            if r.status_code >= 400:
                return JSONResponse(status_code=r.status_code, content={"error": r.text})
            # Diagnostics: return raw text to avoid parser coupling
            return r.text
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.get(ORCH_URL.rstrip("/") + f"/jobs/{job_id}")
            if r.status_code >= 400:
                return JSONResponse(status_code=r.status_code, content={"error": r.text})
            return r.text
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="/app/frontend_dist", html=True), name="static")
# Static assets are mounted at root, AFTER API routes are declared to avoid intercepting /api/*.


