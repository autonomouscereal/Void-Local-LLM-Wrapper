from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs

import httpx
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from fastapi.middleware.cors import CORSMiddleware
from .json_parser import JSONParser
import logging


ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")


def get_engine() -> Engine:
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS conversations (
              id BIGSERIAL PRIMARY KEY,
              title TEXT,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS messages (
              id BIGSERIAL PRIMARY KEY,
              conversation_id BIGINT REFERENCES conversations(id) ON DELETE CASCADE,
              role TEXT,
              content JSONB,
              created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ))
        conn.execute(text(
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
        ))
    return engine


engine = get_engine()
app = FastAPI(title="Chat UI Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount static AFTER API routes to avoid intercepting /api/* with 404/405


@app.middleware("http")
async def global_cors_middleware(request: Request, call_next):
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
    with engine.begin() as conn:
        row = conn.execute(text("INSERT INTO conversations (title) VALUES (:t) RETURNING id"), {"t": title}).first()
    return {"id": row[0], "title": title}


@app.get("/api/conversations")
async def list_conversations():
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 100"))
        data = [{"id": r[0], "title": r[1], "created_at": str(r[2]), "updated_at": str(r[3])} for r in rows]
    return {"data": data}


@app.get("/api/conversations/{cid}/messages")
async def list_messages(cid: int):
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, role, content, created_at FROM messages WHERE conversation_id=:id ORDER BY id ASC"), {"id": cid}).mappings().all()
        return {"data": [dict(r) for r in rows]}


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
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO attachments (conversation_id, name, url, mime) VALUES (:c, :n, :u, :m)"), {"c": conversation_id, "n": file.filename, "u": data.get("url"), "m": file.content_type})
    return {"ok": True, "url": data.get("url"), "name": data.get("name")}


@app.get("/api/conversations/{cid}/attachments")
async def list_attachments(cid: int):
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, name, url, mime FROM attachments WHERE conversation_id=:id ORDER BY id"), {"id": cid}).mappings().all()
        return {"data": [dict(r) for r in rows]}


@app.post("/api/conversations/{cid}/message")
async def add_message(cid: int, body: Dict[str, Any]):
    role = (body or {}).get("role") or "user"
    content = (body or {}).get("content") or ""
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (conversation_id, role, content) VALUES (:c, :r, :x)"), {"c": cid, "r": role, "x": json.dumps({"text": content})})
    return {"ok": True}


def _build_openai_messages(base: List[Dict[str, Any]], attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Reuse attachments implicitly by adding a system hint listing URLs; messages can still pass content arrays
    if not attachments:
        return base
    sys = {"role": "system", "content": "Conversation attachments available:\n" + json.dumps(attachments)}
    return [sys] + base


@app.post("/api/conversations/{cid}/chat")
async def chat(cid: int, request: Request):
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
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (conversation_id, role, content) VALUES (:c, 'user', :x)"), {"c": cid, "x": json.dumps({"text": user_content})})
        atts = conn.execute(text("SELECT name, url, mime FROM attachments WHERE conversation_id=:id"), {"id": cid}).mappings().all()

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
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (conversation_id, role, content) VALUES (:c, 'assistant', :x)"), {"c": cid, "x": json.dumps({"text": content})})
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
async def chat_alt(body: Dict[str, Any]):
    cid = int((body or {}).get("conversation_id") or 0)
    user_content = (body or {}).get("content") or ""
    if not cid:
        return JSONResponse(status_code=400, content={"error": "missing conversation_id"})
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (conversation_id, role, content) VALUES (:c, 'user', :x)"), {"c": cid, "x": json.dumps({"text": user_content})})
        atts = conn.execute(text("SELECT name, url, mime FROM attachments WHERE conversation_id=:id"), {"id": cid}).mappings().all()
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
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (conversation_id, role, content) VALUES (:c, 'assistant', :x)"), {"c": cid, "x": json.dumps({"text": assistant})})
    body = json.dumps(data)
    headers = {
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Expose-Headers": "*",
        "Connection": "close",
        "Content-Length": str(len(body.encode("utf-8"))),
    }
    return Response(content=body, media_type="application/json", headers=headers)


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
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (conversation_id, role, content) VALUES (:c, 'user', :x)"), {"c": cid, "x": json.dumps({"text": user_content})})
        atts = conn.execute(text("SELECT name, url, mime FROM attachments WHERE conversation_id=:id"), {"id": cid}).mappings().all()
    oa_msgs = _build_openai_messages([{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": False}
    async with httpx.AsyncClient(timeout=None) as client:
        rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
        data = rr.json()
    assistant = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (conversation_id, role, content) VALUES (:c, 'assistant', :x)"), {"c": cid, "x": json.dumps({"text": assistant})})
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

