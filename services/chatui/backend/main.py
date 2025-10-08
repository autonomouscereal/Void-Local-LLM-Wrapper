from __future__ import annotations
# WARNING: Do NOT add SQLAlchemy to this service. Use asyncpg with proper pooling and raw SQL only.

import os
import json
from typing import Any, Dict, List, Optional
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")


async def init_pool() -> asyncpg.pool.Pool:
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
# Mount static AFTER API routes to avoid intercepting /api/* with 404/405


logging.basicConfig(level=logging.INFO)


@app.middleware("http")
async def global_cors_middleware(request: Request, call_next):
    logging.info("REQ %s %s", request.method, request.url.path)
    # Preflight short-circuit for any path
    if request.method == "OPTIONS":
        return Response(status_code=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "false",
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Max-Age": "86400",
            "Access-Control-Allow-Private-Network": "true",
            "Connection": "close",
        })
    resp = await call_next(request)
    logging.info("RES %s %s %s", request.method, request.url.path, getattr(resp, 'status_code', ''))
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
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
        return {"data": [{"id": r["id"], "role": r["role"], "content": r["content"], "created_at": str(r["created_at"]) } for r in rows]}


@app.post("/api/upload")
async def upload(conversation_id: int = Form(...), file: UploadFile = File(...)):
    # Forward to orchestrator /upload to get a URL, then store attachment
    try:
        # Read file and send to orchestrator's upload endpoint
        content = await file.read()
        async with httpx.AsyncClient(timeout=120) as client:
            files = {"file": (file.filename, content, file.content_type)}
            r = await client.post(ORCH_URL.rstrip("/") + "/upload", files=files)
            r.raise_for_status()
            data = r.json()
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
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3)", cid, role, {"text": content})
    return {"ok": True}


def _build_openai_messages(base: List[Dict[str, Any]], attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Reuse attachments implicitly by adding a system hint listing URLs; messages can still pass content arrays
    if not attachments:
        return base
    sys = {"role": "system", "content": "Conversation attachments available:\n" + json.dumps(attachments)}
    return [sys] + base


@app.post("/api/conversations/{cid}/chat")
async def chat(cid: int, request: Request, background_tasks: BackgroundTasks):
    logging.info("/api/conversations/%s/chat: start", cid)
    # Accept JSON, raw text, or x-www-form-urlencoded, and parse via custom JSON parser
    ct = (request.headers.get("content-type") or "").lower()
    body: Dict[str, Any] = {}
    parser = JSONParser()
    if "application/json" in ct:
        try:
            # Try FastAPI JSON first; if that fails, repair with custom parser
            body = await request.json()
        except Exception:
            raw_bytes = await request.body()
            decoded = raw_bytes.decode("utf-8", errors="replace")
            try:
                body = parser.method_json_loads(decoded)
            except Exception:
                repaired = parser.attempt_repair(decoded)
                try:
                    body = parser.method_json_loads(repaired)
                except Exception:
                    body = {"content": decoded}
    else:
        raw_bytes = await request.body()
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        if "application/x-www-form-urlencoded" in ct:
            form = parse_qs(raw_text, keep_blank_values=True)
            if "content" in form and len(form["content"]) > 0:
                candidate = form["content"][0]
                try:
                    body = parser.method_json_loads(candidate)
                except Exception:
                    repaired = parser.attempt_repair(candidate)
                    try:
                        body = parser.method_json_loads(repaired)
                    except Exception:
                        body = {"content": candidate}
            elif len(form) == 1:
                first_key = next(iter(form))
                vals = form.get(first_key) or [raw_text]
                candidate = vals[0]
                try:
                    body = parser.method_json_loads(candidate)
                except Exception:
                    repaired = parser.attempt_repair(candidate)
                    try:
                        body = parser.method_json_loads(repaired)
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
    # Force non-stream proxy for reliability in the UI
    stream = False
    # Store user message
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2)", cid, {"text": user_content})
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)

    # Build messages for orchestrator
    oa_msgs = _build_openai_messages(messages + [{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}

    async def _proxy_stream():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                url = ORCH_URL.rstrip("/") + "/v1/chat/completions"
                async with client.stream("POST", url, json=payload) as r:
                    ct = r.headers.get("content-type", "")
                    if stream and ct.startswith("text/event-stream"):
                        async for chunk in r.aiter_text():
                            # pass through SSE chunks unchanged
                            yield chunk
                    else:
                        body = await r.aread()
                        # ensure JSON text
                        try:
                            decoded = body.decode("utf-8")
                            obj = parser.method_json_loads(decoded)
                            yield json.dumps(obj)
                        except Exception:
                            try:
                                repaired = parser.attempt_repair(decoded)
                                obj = parser.method_json_loads(repaired)
                                yield json.dumps(obj)
                            except Exception:
                                yield body
        except Exception as ex:
            yield json.dumps({"error": str(ex)})

    # Non-stream: forward and persist assistant message
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
            if rr.status_code >= 400:
                content_type = rr.headers.get("content-type", "")
                if content_type.startswith("application/json"):
                    logging.warning("orchestrator error %s: %s", rr.status_code, rr.text[:500])
                    return JSONResponse(status_code=rr.status_code, content=rr.json())
                logging.warning("orchestrator error %s (non-json): %s", rr.status_code, rr.text[:500])
                return JSONResponse(status_code=rr.status_code, content={"error": rr.text})
            try:
                data = rr.json()
            except Exception:
                parser = JSONParser()
                decoded = rr.text
                try:
                    data = parser.method_json_loads(decoded)
                except Exception:
                    repaired = parser.attempt_repair(decoded)
                    data = parser.method_json_loads(repaired)
    except Exception as ex:
        logging.exception("chat proxy failed: %s", ex)
        return JSONResponse(status_code=502, content={"error": str(ex)})
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    async def _persist_assistant_message(conv_id: int, text_value: str) -> None:
        try:
            async with _pool().acquire() as c2:
                await c2.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2)", conv_id, {"text": text_value})
        except Exception:
            logging.exception("persist assistant failed cid=%s", conv_id)
    background_tasks.add_task(_persist_assistant_message, cid, content)
    # Return explicit body with content-length to avoid client transport quirks
    body = json.dumps(data)
    headers = {
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Expose-Headers": "*",
        "Connection": "close",
        "Content-Length": str(len(body.encode("utf-8"))),
    }
    logging.info("/api/conversations/%s/chat: done", cid)
    return Response(content=body, media_type="application/json", headers=headers)


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
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2)", cid, {"text": user_content})
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)
    oa_msgs = _build_openai_messages([{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}
    async with httpx.AsyncClient(timeout=None) as client:
        rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
        try:
            data = rr.json()
        except Exception:
            parser = JSONParser()
            decoded = rr.text
            try:
                data = parser.method_json_loads(decoded)
            except Exception:
                repaired = parser.attempt_repair(decoded)
                data = parser.method_json_loads(repaired)
    assistant = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    async def _persist_assistant_message(conv_id: int, text_value: str) -> None:
        try:
            async with _pool().acquire() as c2:
                await c2.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2)", conv_id, {"text": text_value})
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
    logging.info("/api/chat: done cid=%s", cid)
    return Response(content=body, media_type="application/json", headers=headers)


# Neutral path versions (avoid filters on "chat")
@app.post("/api/conversations/{cid}/call")
async def call_conv(cid: int, request: Request, background_tasks: BackgroundTasks):
    logging.info("/api/conversations/%s/call: start", cid)
    parser = JSONParser()
    try:
        payload_in = await request.json()
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
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2)", cid, {"text": user_content})
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)
    oa_msgs = _build_openai_messages([{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}
    async with httpx.AsyncClient(timeout=None) as client:
        rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
        try:
            data = rr.json()
        except Exception:
            decoded = rr.text
            try:
                data = parser.method_json_loads(decoded)
            except Exception:
                repaired = parser.attempt_repair(decoded)
                data = parser.method_json_loads(repaired)
    assistant = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    async def _persist_assistant_message(conv_id: int, text_value: str) -> None:
        try:
            async with _pool().acquire() as c2:
                await c2.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2)", conv_id, {"text": text_value})
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
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2)", cid, {"text": user_content})
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)
    oa_msgs = _build_openai_messages([{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}
    parser = JSONParser()
    async with httpx.AsyncClient(timeout=None) as client:
        rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
        try:
            data = rr.json()
        except Exception:
            decoded = rr.text
            try:
                data = parser.method_json_loads(decoded)
            except Exception:
                repaired = parser.attempt_repair(decoded)
                data = parser.method_json_loads(repaired)
    assistant = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    async def _persist_assistant_message(conv_id: int, text_value: str) -> None:
        try:
            async with _pool().acquire() as c2:
                await c2.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2)", conv_id, {"text": text_value})
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
    logging.info("/api/call: done cid=%s", cid)
    return Response(content=body, media_type="application/json", headers=headers)


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
        async with httpx.AsyncClient(timeout=120) as client:
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
        async with httpx.AsyncClient(timeout=10) as client:
            h = await client.get(ORCH_URL.rstrip("/") + "/healthz")
            out["healthz_status"] = h.status_code
            out["healthz_body"] = (h.json() if "application/json" in (h.headers.get("content-type") or "") else h.text)
    except Exception as ex:
        out["healthz_error"] = str(ex)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            d = await client.get(ORCH_URL.rstrip("/") + "/debug")
            out["debug_status"] = d.status_code
            out["debug_body"] = (d.json() if "application/json" in (d.headers.get("content-type") or "") else d.text)
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
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'user', $2)", cid, json.dumps({"text": user_content}))
        atts = await conn.fetch("SELECT name, url, mime FROM attachments WHERE conversation_id=$1", cid)
    oa_msgs = _build_openai_messages([{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}
    async with httpx.AsyncClient(timeout=None) as client:
        rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
        data = rr.json()
    assistant = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    async with _pool().acquire() as conn:
        await conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, 'assistant', $2)", cid, json.dumps({"text": assistant}))
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
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(ORCH_URL.rstrip("/") + "/jobs", params=params)
            if r.status_code >= 400:
                return JSONResponse(status_code=r.status_code, content={"error": r.text})
            return r.json()
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(ORCH_URL.rstrip("/") + f"/jobs/{job_id}")
            if r.status_code >= 400:
                return JSONResponse(status_code=r.status_code, content={"error": r.text})
            return r.json()
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="/app/frontend_dist", html=True), name="static")

