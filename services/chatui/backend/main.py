from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from fastapi.middleware.cors import CORSMiddleware


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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount static AFTER API routes to avoid intercepting /api/* with 404/405


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


def _build_openai_messages(base: List[Dict[str, Any]], attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Reuse attachments implicitly by adding a system hint listing URLs; messages can still pass content arrays
    if not attachments:
        return base
    sys = {"role": "system", "content": "Conversation attachments available:\n" + json.dumps(attachments)}
    return [sys] + base


@app.post("/api/conversations/{cid}/chat")
async def chat(cid: int, body: Dict[str, Any]):
    user_content = (body or {}).get("content") or ""
    messages = (body or {}).get("messages") or []
    stream = bool((body or {}).get("stream"))
    # Store user message
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (conversation_id, role, content) VALUES (:c, 'user', :x)"), {"c": cid, "x": json.dumps({"text": user_content})})
        atts = conn.execute(text("SELECT name, url, mime FROM attachments WHERE conversation_id=:id"), {"id": cid}).mappings().all()

    # Build messages for orchestrator
    oa_msgs = _build_openai_messages(messages + [{"role": "user", "content": user_content}], [dict(a) for a in atts])
    payload = {"messages": oa_msgs, "stream": stream}

    async def _proxy_stream():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                r = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
                if stream and r.headers.get("content-type", "").startswith("text/event-stream"):
                    async for chunk in r.aiter_text():
                        yield chunk
                else:
                    data = r.json()
                    yield json.dumps(data)
        except Exception as ex:
            yield json.dumps({"error": str(ex)})

    if stream:
        return StreamingResponse(_proxy_stream(), media_type="text/event-stream")
    else:
        # Non-stream: forward and persist assistant message
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                rr = await client.post(ORCH_URL.rstrip("/") + "/v1/chat/completions", json=payload)
                rr.raise_for_status()
                data = rr.json()
        except Exception as ex:
            return JSONResponse(status_code=500, content={"error": str(ex)})
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO messages (conversation_id, role, content) VALUES (:c, 'assistant', :x)"), {"c": cid, "x": json.dumps({"text": content})})
        return data


@app.get("/api/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    params = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(ORCH_URL.rstrip("/") + "/jobs", params=params)
            r.raise_for_status()
            return r.json()
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(ORCH_URL.rstrip("/") + f"/jobs/{job_id}")
            r.raise_for_status()
            return r.json()
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="/app/frontend_dist", html=True), name="static")

