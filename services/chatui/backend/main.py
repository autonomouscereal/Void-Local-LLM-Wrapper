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

import anyio
import httpx
from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
import asyncpg
from fastapi.middleware.cors import CORSMiddleware
from .json_parser import JSONParser
import logging
import websockets  # type: ignore

# Single module logger.
log = logging.getLogger(__name__)

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

# Reuse keep-alive connections to the orchestrator to avoid per-request TCP churn.
_ORCH_PROXY_TRANSPORT = httpx.AsyncHTTPTransport(
    retries=0,
    http2=False,
    limits=httpx.Limits(max_connections=32, max_keepalive_connections=32, keepalive_expiry=60.0),
)
_ORCH_PROXY_CLIENT = httpx.AsyncClient(
    trust_env=False,
    timeout=None,
    base_url=ORCH_URL.rstrip("/"),
    transport=_ORCH_PROXY_TRANSPORT,
)


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
              orchestrator_conversation_id TEXT,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        # Forward-compatible: if the table already existed, add orchestrator_conversation_id without needing a migration.
        try:
            await conn.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS orchestrator_conversation_id TEXT;")
        except Exception:
            # Non-fatal (column may already exist), but never swallow silently.
            log.warning("db.init: ALTER TABLE conversations ADD COLUMN orchestrator_conversation_id failed", exc_info=True)
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
    """
    Decode DB JSON-ish fields into a stable dict shape.

    Always delegate parsing to the global JSONParser; do not pre-screen on
    leading characters. This lets the repair logic handle messy strings while
    still falling back cleanly to a {"text": ...} wrapper when needed.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parser = JSONParser()
        try:
            expected = {"text": str}
            obj = parser.parse(value, expected)
            if isinstance(obj, dict) and obj:
                return obj
        except Exception as exc:
            # Fall through to plain-text wrapper below, but do not be silent.
            logging.getLogger(__name__).warning("decode_json failed; falling back to text wrapper: %s", exc, exc_info=True)
        return {"text": value}
    return {"text": str(value or "")}

# Mount static AFTER API routes to avoid intercepting /api/* with 404/405


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
#    - Added explicit OPTIONS endpoints for /api/chat and /api/conversations/{conversation_id}/chat.
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
        hdrs = {
            "Access-Control-Allow-Origin": (request.headers.get("origin") or "*"),
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": (request.headers.get("access-control-request-headers") or "*"),
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Allow-Private-Network": "true",
            "Connection": "close",
            "Vary": "Origin",
            "Content-Length": "0",
        }
        return Response(status_code=200, headers=hdrs)
    resp = await call_next(request)
    logging.info("RES %s %s %s", request.method, request.url.path, getattr(resp, "status_code", ""))
    # Inject permissive CORS headers on every response
    origin = request.headers.get("origin") or "*"
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Credentials"] = "true"
    resp.headers["Access-Control-Expose-Headers"] = "*"
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
        row = await conn.fetchrow(
            "INSERT INTO conversations (title, orchestrator_conversation_id) VALUES ($1, $2) RETURNING id, orchestrator_conversation_id",
            title,
            None,
        )
    return {"id": row[0], "title": title, "orchestrator_conversation_id": row[1]}


@app.get("/api/conversations")
async def list_conversations():
    async with _pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, orchestrator_conversation_id, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 100"
        )
        data = [
            {
                "id": r["id"],
                "title": r["title"],
                "orchestrator_conversation_id": r["orchestrator_conversation_id"],
                "created_at": str(r["created_at"]),
                "updated_at": str(r["updated_at"]),
            }
            for r in rows
        ]
    return {"data": data}


@app.post("/api/conversations/{conversation_id}/orchestrator_conversation_id")
async def set_orchestrator_conversation_id(conversation_id: int, body: Dict[str, Any]):
    orchestrator_conversation_id = (body or {}).get("orchestrator_conversation_id")
    orchestrator_conversation_id = (
        str(orchestrator_conversation_id).strip() if orchestrator_conversation_id is not None else ""
    )
    if not orchestrator_conversation_id:
        return JSONResponse(status_code=400, content={"error": "missing orchestrator_conversation_id"})
    async with _pool().acquire() as conn:
        await conn.execute(
            "UPDATE conversations SET orchestrator_conversation_id=$1, updated_at=NOW() WHERE id=$2",
            orchestrator_conversation_id,
            conversation_id,
        )
    return {"ok": True, "orchestrator_conversation_id": orchestrator_conversation_id}


@app.get("/api/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: int):
    async with _pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, role, content, created_at FROM messages WHERE conversation_id=$1 ORDER BY id ASC",
            conversation_id,
        )
        return {
            "data": [
                {"id": r["id"], "role": r["role"], "content": _decode_json(r["content"]), "created_at": str(r["created_at"])}
                for r in rows
            ]
        }


@app.post("/api/upload")
async def upload(conversation_id: int = Form(...), file: UploadFile = File(...)):
    return JSONResponse(status_code=410, content={"error": "proxy disabled; use orchestrator /upload directly"})


@app.post("/api/attachments.add")
async def attachments_add(body: Dict[str, Any]):
    conversation_id = int((body or {}).get("conversation_id") or 0)
    attachment_name = (body or {}).get("attachment_name") or ""
    if not attachment_name and isinstance(body, dict) and "name" in body and isinstance(body["name"], str):
        attachment_name = body["name"]
    url = (body or {}).get("url") or ""
    mime = (body or {}).get("mime") or "application/octet-stream"
    if not conversation_id or not url:
        return JSONResponse(status_code=400, content={"error": "missing conversation_id or url"})
    async with _pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO attachments (conversation_id, name, url, mime) VALUES ($1, $2, $3, $4)",
            conversation_id,
            attachment_name,
            url,
            mime,
        )
    return {"ok": True}


@app.get("/api/conversations/{conversation_id}/attachments")
async def list_attachments(conversation_id: int):
    async with _pool().acquire() as conn:
        rows = await conn.fetch("SELECT id, name, url, mime FROM attachments WHERE conversation_id=$1 ORDER BY id", conversation_id)
        return {"data": [{"attachment_id": r["id"], "attachment_name": r["name"], "url": r["url"], "mime": r["mime"] } for r in rows]}


@app.post("/api/conversations/{conversation_id}/message")
async def add_message(conversation_id: int, body: Dict[str, Any]):
    role = (body or {}).get("role") or "user"
    content = (body or {}).get("content") or ""
    async with _pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3::jsonb)",
            conversation_id,
            role,
            json.dumps({"text": content}),
        )
    return {"ok": True}


def _build_openai_messages(base: List[Dict[str, Any]], attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Reuse attachments implicitly by adding a system hint listing URLs; messages can still pass content arrays
    if not attachments:
        return base
    sys = {"role": "system", "content": "Conversation attachments available:\n" + json.dumps(attachments)}
    return [sys] + base





@app.options("/api/{path:path}")
async def any_preflight(path: str):
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Private-Network": "true",
    })


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







@app.options("/api/chat")
async def chat_alt_preflight():
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Private-Network": "true",
    })





@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/config.js")
async def config_js(request: Request):
    # Prefer explicit public base if provided; otherwise derive from request host with port 8000
    base = os.getenv("PUBLIC_ORCH_BASE", "").strip()
    if not base:
        host = (request.headers.get("host") or "").split(":")[0]
        scheme = "https" if str(request.url.scheme).lower() == "https" else "http"
        base = f"{scheme}://{host}:8000"
    base = base.rstrip("/")
    body = f"window.__ORCH_BASE__ = {json.dumps(base)};\n"
    headers = {
        "Cache-Control": "no-store",
        "Content-Type": "application/javascript; charset=utf-8",
        "Content-Length": str(len(body.encode("utf-8"))),
        "Connection": "close",
    }
    return Response(content=body, media_type="application/javascript", headers=headers)

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


# Same-origin proxy for chat completions to avoid browser-side CORS/network failures.
@app.post("/api/orch/v1/chat/completions")
async def orch_chat_completions_proxy(request: Request):
    # Proxy-only streaming keepalive:
    # - We do NOT stream the real chat completion.
    # - We DO send harmless whitespace bytes periodically so the browser connection stays open
    #   while the upstream orchestrator/Ollama call blocks.
    try:
        raw = await request.body()
        done = anyio.Event()
        result: Dict[str, Any] = {"body": b"", "status": 200, "content_type": "application/json; charset=utf-8"}

        async def fetch_upstream():
            nonlocal result
            r = await _ORCH_PROXY_CLIENT.post(
                "/v1/chat/completions",
                content=raw,
                headers={
                    # Body is still JSON text; orchestrator accepts/decodes from raw bytes.
                    "Content-Type": "text/plain; charset=utf-8",
                    "Accept": request.headers.get("accept") or "*/*",
                },
            )
            try:
                body_bytes = await r.aread()
            finally:
                await r.aclose()
            result = {
                "body": body_bytes,
                "status": int(getattr(r, "status_code", 200) or 200),
                "content_type": r.headers.get("content-type") or "application/json; charset=utf-8",
            }
            done.set()

        async def gen():
            async with anyio.create_task_group() as tg:
                tg.start_soon(fetch_upstream)
                # Send at least one byte immediately so headers flush and the socket is "active".
                yield b" "
                while not done.is_set():
                    await anyio.sleep(1.0)
                    yield b" "
            # Upstream finished; now send the real JSON body (whitespace prefix is valid JSON).
            yield result.get("body") or b""

        # Browser-facing response: streaming (chunked). Do not set Content-Length.
        headers = {
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Private-Network": "true",
            "Access-Control-Expose-Headers": "Content-Type",
        }
        return StreamingResponse(gen(), status_code=200, media_type="application/json", headers=headers)
    except Exception as ex:
        return JSONResponse(status_code=502, content={"error": "proxy failure", "detail": str(ex)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/minimal")
async def minimal_ui(request: Request):
    """
    Minimal, no-framework UI that POSTS directly to the orchestrator /v1/chat/completions.
    No proxies, no polling, no extras. Use this to isolate browser/network issues.
    """
    base = os.getenv("PUBLIC_ORCH_BASE", "").strip()
    if not base:
        host = (request.headers.get("host") or "").split(":")[0]
        scheme = "https" if str(request.url.scheme).lower() == "https" else "http"
        base = f"{scheme}://{host}:8000"
    base = base.rstrip("/")
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Minimal Completions Client</title>
    <style>
      body {{ background:#0b0b0f; color:#e6e6e6; font-family: ui-sans-serif, system-ui, Arial; margin:0; padding:20px; }}
      .row {{ display:flex; gap:8px; margin-bottom:8px; align-items:center; }}
      input, textarea, button {{ background:#0f172a; color:#fff; border:1px solid #333; border-radius:6px; padding:8px; }}
      textarea {{ width:100%; }}
      #out {{ white-space:pre-wrap; background:#111827; border:1px solid #222; border-radius:8px; padding:12px; min-height:160px; }}
      label.small {{ font-size:12px; color:#9ca3af; }}
    </style>
  </head>
  <body>
    <h2>Minimal /v1/chat/completions Client</h2>
    <div class="row">
      <label class="small">ORCH_BASE</label>
      <input id="base" value="{base}" style="flex:1" />
      <button id="ping">Ping</button>
    </div>
    <div class="row">
      <textarea id="prompt" rows="4" placeholder="Describe the image you want...">please draw me a picture of shadow the hedgehog</textarea>
    </div>
    <div class="row">
      <button id="send">Send</button>
      <label class="small" id="status"></label>
    </div>
    <div id="out"></div>
    <script>
      const statusEl = document.getElementById('status');
      const outEl = document.getElementById('out');
      const baseEl = document.getElementById('base');
      const promptEl = document.getElementById('prompt');

      function setStatus(s) {{ statusEl.textContent = s; }}
      function setOut(t) {{ outEl.textContent = t; }}

      document.getElementById('ping').onclick = async () => {{
        const u = baseEl.value.replace(/\\/$/, '') + '/healthz';
        setStatus('Pinging ' + u + ' ...');
        try {{
          const r = await fetch(u, {{ mode: 'cors', credentials: 'omit' }});
          const txt = await r.text();
          setOut('PING ' + r.status + '\\n' + txt);
          setStatus('OK');
        }} catch (e) {{
          setOut('PING failed: ' + (e && e.message || e));
          setStatus('ERROR');
        }}
      }};

      document.getElementById('send').onclick = async () => {{
        const ORCH_BASE = baseEl.value.replace(/\\/$/, '');
        const url = ORCH_BASE + '/v1/chat/completions';
        const body = JSON.stringify({{ messages: [{{ role: 'user', content: promptEl.value }}], stream: false }});
        setStatus('Sending...');
        setOut('');
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url, true);
        xhr.timeout = null; // no client-side timeout
        xhr.withCredentials = false;
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.setRequestHeader('Accept', '*/*');
        xhr.onreadystatechange = () => {{
          if (xhr.readyState !== 4) return;
          setStatus('Done (' + xhr.status + ')');
          const txt = xhr.responseText || '';
          try {{
            const obj = txt ? JSON.parse(txt) : null;
            setOut(JSON.stringify(obj, null, 2));
          }} catch {{
            setOut(txt);
          }}
        }};
        xhr.onabort = () => {{ setStatus('Aborted'); setOut('Request aborted by browser'); }};
        xhr.onerror = () => {{ setStatus('Network error'); setOut('Network error'); }};
        try {{ xhr.send(body); }} catch (e) {{
          setStatus('Send failed');
          setOut('Send failed: ' + (e && e.message || e));
        }}
      }};
    </script>
  </body>
</html>
"""
    headers = {
        "Cache-Control": "no-store",
        "Content-Type": "text/html; charset=utf-8",
        "Content-Length": str(len(html.encode("utf-8"))),
        "Connection": "close",
    }
    return Response(content=html, media_type="text/html", headers=headers)




@app.websocket("/api/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.close(code=1008)


app.mount("/", StaticFiles(directory="/app/frontend_dist", html=True), name="static")
# Serve the shared artifacts volume (same one orchestrator writes into).
# This makes `/uploads/...` work even when the browser is talking to ChatUI
# as the origin (e.g., `http://host:3000/uploads/...`), and keeps URLs
# consistent with orchestrator-emitted paths.
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except Exception:
    # Keep the service up even if the shared volume is unavailable.
    log.error("UPLOAD_DIR create failed path=%r; falling back to cwd", UPLOAD_DIR, exc_info=True)
    UPLOAD_DIR = "."
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
# Static assets are mounted at root, AFTER API routes are declared to avoid intercepting /api/*. 


