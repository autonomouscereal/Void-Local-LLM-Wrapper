from __future__ import annotations

import os
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
import time

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11434")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen2.5:32b-instruct-q4_K_M")
GPTOSS_BASE_URL = os.getenv("GPTOSS_BASE_URL", "http://localhost:11435")
GPTOSS_MODEL_ID = os.getenv("GPTOSS_MODEL_ID", "gpt-oss:20b-q5_K_M")
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "8192"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
ENABLE_WEBSEARCH = os.getenv("ENABLE_WEBSEARCH", "false").lower() == "true"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
MCP_HTTP_BRIDGE_URL = os.getenv("MCP_HTTP_BRIDGE_URL")  # e.g., http://host.docker.internal:9999
EXECUTOR_BASE_URL = os.getenv("EXECUTOR_BASE_URL")  # http://executor:8081
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen")  # qwen | gptoss
ENABLE_DEBATE = os.getenv("ENABLE_DEBATE", "true").lower() == "true"
MAX_DEBATE_TURNS = int(os.getenv("MAX_DEBATE_TURNS", "1"))
ALLOW_TOOL_EXECUTION = os.getenv("ALLOW_TOOL_EXECUTION", "true").lower() == "true"
AUTO_EXECUTE_TOOLS = os.getenv("AUTO_EXECUTE_TOOLS", "true").lower() == "true"

# RAG configuration (pgvector)
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RAG_CACHE_TTL_SEC = int(os.getenv("RAG_CACHE_TTL_SEC", "300"))


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    user: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    finish_reason: Optional[str] = None
    message: Dict[str, Any]


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[ChatChoice]
    model: str
    usage: Optional[Dict[str, int]] = None


app = FastAPI(title="Void Orchestrator", version="0.1.0")


def build_ollama_payload(messages: List[ChatMessage], model: str, num_ctx: int, temperature: float) -> Dict[str, Any]:
    rendered: List[str] = []
    for m in messages:
        if m.role == "tool":
            tool_name = m.name or "tool"
            rendered.append(f"tool[{tool_name}]: {m.content}")
        else:
            content = m.content if m.content is not None else ""
            rendered.append(f"{m.role}: {content}")
    prompt = "\n".join(rendered)
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
        },
    }


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # crude approximation: 1 token ~ 4 chars
    return max(1, (len(text) + 3) // 4)


def estimate_usage(messages: List[ChatMessage], completion_text: str) -> Dict[str, int]:
    prompt_text = "\n".join([(m.content or "") for m in messages])
    prompt_tokens = estimate_tokens_from_text(prompt_text)
    completion_tokens = estimate_tokens_from_text(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def merge_usages(usages: List[Optional[Dict[str, int]]]) -> Dict[str, int]:
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usages:
        if not u:
            continue
        out["prompt_tokens"] += int(u.get("prompt_tokens", 0))
        out["completion_tokens"] += int(u.get("completion_tokens", 0))
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


# ---------- RAG (pgvector) ----------
_engine: Optional[Engine] = None
_embedder: Optional[SentenceTransformer] = None
_rag_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def get_engine() -> Optional[Engine]:
    global _engine
    if _engine is not None:
        return _engine
    if not (POSTGRES_HOST and POSTGRES_DB and POSTGRES_USER and POSTGRES_PASSWORD):
        return None
    url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    _engine = create_engine(url, pool_pre_ping=True)
    with _engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS rag_docs (
              id BIGSERIAL PRIMARY KEY,
              path TEXT,
              chunk TEXT,
              embedding vector(384)
            );
            """
        ))
        conn.execute(text("CREATE INDEX IF NOT EXISTS rag_docs_embedding_idx ON rag_docs USING ivfflat (embedding vector_cosine) WITH (lists = 100);"))
    return _engine


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


async def rag_index_dir(root: str = "/workspace", glob_exts: Optional[List[str]] = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    import glob as _glob
    import os as _os
    engine = get_engine()
    if engine is None:
        return {"error": "pgvector not configured"}
    exts = glob_exts or ["*.md", "*.py", "*.ts", "*.tsx", "*.js", "*.json", "*.txt"]
    files: List[str] = []
    for ext in exts:
        files.extend(_glob.glob(_os.path.join(root, "**", ext), recursive=True))
    embedder = get_embedder()
    total_chunks = 0
    with engine.begin() as conn:
        for fp in files[:5000]:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
            i = 0
            length = len(content)
            while i < length:
                chunk = content[i : i + chunk_size]
                i += max(1, chunk_size - chunk_overlap)
                vec = embedder.encode([chunk])[0]
                conn.execute(text("INSERT INTO rag_docs (path, chunk, embedding) VALUES (:p, :c, :e)"), {"p": fp.replace(root + "/", ""), "c": chunk, "e": list(vec)})
                total_chunks += 1
    return {"indexed_files": len(files), "chunks": total_chunks}


async def rag_search(query: str, k: int = 8) -> List[Dict[str, Any]]:
    now = time.time()
    key = f"{query}::{k}"
    cached = _rag_cache.get(key)
    if cached and (now - cached[0] <= RAG_CACHE_TTL_SEC):
        return cached[1]
    engine = get_engine()
    if engine is None:
        return []
    vec = get_embedder().encode([query])[0]
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT path, chunk FROM rag_docs ORDER BY embedding <=> :q LIMIT :k"), {"q": list(vec), "k": k}).fetchall()
        results = [{"path": r[0], "chunk": r[1]} for r in rows]
        _rag_cache[key] = (now, results)
        return results


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=2))
async def call_ollama(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            resp = await client.post(f"{base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and ("prompt_eval_count" in data or "eval_count" in data):
                usage = {
                    "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
                    "completion_tokens": int(data.get("eval_count", 0) or 0),
                }
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                data["_usage"] = usage
            return data
        except httpx.HTTPError as e:
            return {"error": str(e), "_base_url": base_url}


async def serpapi_google_search(queries: List[str], max_results: int = 5) -> str:
    if not SERPAPI_API_KEY:
        return ""
    aggregate = []
    async with httpx.AsyncClient(timeout=30) as client:
        for q in queries:
            try:
                params = {
                    "engine": "google",
                    "q": q,
                    "num": max_results,
                    "api_key": SERPAPI_API_KEY,
                }
                r = await client.get("https://serpapi.com/search.json", params=params)
                r.raise_for_status()
                data = r.json()
                organic = data.get("organic_results", [])
                for item in organic[:max_results]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    aggregate.append(f"- {title}\n{snippet}\n{link}")
            except Exception:
                continue
    return "\n".join(aggregate[: max_results * max(1, len(queries))])


async def call_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not MCP_HTTP_BRIDGE_URL:
        return {"error": "MCP bridge URL not configured"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                MCP_HTTP_BRIDGE_URL.rstrip("/") + "/call",
                json={"name": name, "arguments": arguments or {}},
            )
            r.raise_for_status()
            return r.json()
    except Exception as ex:
        return {"error": str(ex)}


def to_openai_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for idx, c in enumerate(tool_calls):
        name = c.get("name") or "tool"
        args = c.get("arguments") or {}
        converted.append({
            "id": f"call_{idx+1}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return converted


async def propose_search_queries(messages: List[ChatMessage]) -> List[str]:
    guidance = (
        "Given the conversation, propose up to 3 concise web search queries that would most improve the answer. "
        "Return each query on its own line with no extra text."
    )
    prompt_messages = messages + [ChatMessage(role="user", content=guidance)]
    payload = build_ollama_payload(prompt_messages, QWEN_MODEL_ID, DEFAULT_NUM_CTX, DEFAULT_TEMPERATURE)
    try:
        result = await call_ollama(QWEN_BASE_URL, payload)
        text = result.get("response", "")
        lines = [ln.strip("- ") for ln in text.splitlines() if ln.strip()]
        # take up to 3 non-empty queries
        return lines[:3]
    except Exception:
        return []


def meta_prompt(messages: List[ChatMessage]) -> List[ChatMessage]:
    system_preface = (
        "You are part of a two-model team with explicit roles: a Planner and Executors. "
        "Planner decomposes the task, chooses tools, and requests relevant evidence. Executors produce solutions and critiques. "
        "Be precise, show working when non-trivial, and output correct, runnable code when appropriate."
    )
    return [ChatMessage(role="system", content=system_preface)] + messages


def merge_responses(request: ChatRequest, qwen_text: str, gptoss_text: str) -> str:
    return (
        "Committee Synthesis:\n\n"
        "Qwen perspective:\n" + qwen_text.strip() + "\n\n"
        "GPT-OSS perspective:\n" + gptoss_text.strip() + "\n\n"
        "Final answer (synthesize both; prefer correctness and specificity):\n"
    )


def build_tools_section(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return ""
    try:
        return "Available tools (JSON schema):\n" + json.dumps(tools, indent=2)
    except Exception:
        return ""


async def planner_produce_plan(messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]], temperature: float) -> Tuple[str, List[Dict[str, Any]]]:
    planner_id = QWEN_MODEL_ID if PLANNER_MODEL.lower() == "qwen" else GPTOSS_MODEL_ID
    planner_base = QWEN_BASE_URL if PLANNER_MODEL.lower() == "qwen" else GPTOSS_BASE_URL
    guide = (
        "You are the Planner. Produce a short step-by-step plan and, if helpful, propose up to 3 tool invocations.\n"
        "Return strict JSON with keys: plan (string), tool_calls (array of {name: string, arguments: object})."
    )
    tool_info = build_tools_section(tools)
    plan_messages = messages + [ChatMessage(role="user", content=guide + ("\n" + tool_info if tool_info else ""))]
    payload = build_ollama_payload(plan_messages, planner_id, DEFAULT_NUM_CTX, temperature)
    result = await call_ollama(planner_base, payload)
    text = result.get("response", "").strip()
    json_text = text
    if "```" in text:
        # try to extract code block
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            candidate = parts[i]
            if candidate.lstrip().startswith("{"):
                json_text = candidate
                break
    plan = ""
    tool_calls: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(json_text)
        plan = parsed.get("plan", "")
        tool_calls = parsed.get("tool_calls", []) or []
    except Exception:
        plan = text
        tool_calls = []
    return plan, tool_calls


async def execute_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name")
    args = call.get("arguments") or {}
    if name == "web_search" and ENABLE_WEBSEARCH and SERPAPI_API_KEY and ALLOW_TOOL_EXECUTION:
        query = args.get("query") or ""
        if not query:
            return {"name": name, "error": "missing query"}
        snippets = await serpapi_google_search([query], max_results=int(args.get("k", 5)))
        return {"name": name, "result": snippets}
    if name == "rag_index":
        res = await rag_index_dir(root=args.get("root", "/workspace"))
        return {"name": name, "result": res}
    if name == "rag_search":
        query = args.get("query")
        k = int(args.get("k", 8))
        if not query:
            return {"name": name, "error": "missing query"}
        res = await rag_search(query, k)
        return {"name": name, "result": res}
    if name == "run_python" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/run_python", json={"code": args.get("code", "")})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "write_file" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/write_file", json={"path": args.get("path"), "content": args.get("content", "")})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    if name == "read_file" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/read_file", json={"path": args.get("path")})
            try:
                r.raise_for_status()
                return {"name": name, "result": r.json()}
            except Exception:
                return {"name": name, "error": r.text}
    # MCP tool forwarding (HTTP bridge)
    if name and (name.startswith("mcp:") or args.get("mcpTool") is True) and ALLOW_TOOL_EXECUTION:
        res = await call_mcp_tool(name.replace("mcp:", "", 1), args)
        return {"name": name, "result": res}
    # Unknown tool - return as unexecuted
    return {"name": name or "unknown", "skipped": True, "reason": "unsupported tool in orchestrator"}


async def execute_tools(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for call in tool_calls[:5]:
        try:
            res = await execute_tool_call(call)
            results.append(res)
        except Exception as ex:
            results.append({"name": call.get("name", "unknown"), "error": str(ex)})
    return results


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest, request: Request):
    messages = meta_prompt(body.messages)

    # If client supplies tool results (role=tool) we include them verbatim for the planner/executors

    # Optional self-ask-with-search augmentation
    if ENABLE_WEBSEARCH and SERPAPI_API_KEY:
        queries = await propose_search_queries(messages)
        if queries:
            snippets = await serpapi_google_search(queries, max_results=5)
            if snippets:
                messages = [
                    ChatMessage(role="system", content="Web search results follow. Use them only if relevant."),
                    ChatMessage(role="system", content=snippets),
                ] + messages

    # 1) Planner proposes plan + tool calls
    plan_text, tool_calls = await planner_produce_plan(messages, body.tools, body.temperature or DEFAULT_TEMPERATURE)
    # tool_choice=required compatibility: force at least one tool_call if tools are provided
    if (body.tool_choice == "required") and (not tool_calls) and body.tools:
        # choose first declared tool with empty args
        first = body.tools[0]
        fn = (first.get("function") or {}).get("name") or first.get("name") or "tool"
        tool_calls = [{"name": fn, "arguments": {}}]

    # If tool semantics are client-driven, return tool_calls instead of executing
    if tool_calls and not AUTO_EXECUTE_TOOLS:
        tool_calls_openai = to_openai_tool_calls(tool_calls)
        response = {
            "id": "orc-1",
            "object": "chat.completion",
            "model": f"committee:{QWEN_MODEL_ID}+{GPTOSS_MODEL_ID}",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_openai,
                    },
                }
            ],
        }
        if body.stream:
            async def _stream_once():
                chunk = json.dumps({"id": response["id"], "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": tool_calls_openai}, "finish_reason": None}]})
                yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream_once(), media_type="text/event-stream")
        # include usage estimate even in tool_calls path (no completion tokens yet)
        response["usage"] = estimate_usage(messages, "")
        return JSONResponse(content=response)

    # 2) Optionally execute tools
    tool_results: List[Dict[str, Any]] = []
    if tool_calls:
        tool_results = await execute_tools(tool_calls)

    # 3) Executors respond independently using plan + evidence
    evidence_blocks: List[ChatMessage] = []
    if plan_text:
        evidence_blocks.append(ChatMessage(role="system", content=f"Planner plan:\n{plan_text}"))
    if tool_results:
        evidence_blocks.append(ChatMessage(role="system", content="Tool results:\n" + json.dumps(tool_results, indent=2)))
    exec_messages = evidence_blocks + messages

    qwen_payload = build_ollama_payload(
        messages=exec_messages, model=QWEN_MODEL_ID, num_ctx=DEFAULT_NUM_CTX, temperature=body.temperature or DEFAULT_TEMPERATURE
    )
    gptoss_payload = build_ollama_payload(
        messages=exec_messages, model=GPTOSS_MODEL_ID, num_ctx=DEFAULT_NUM_CTX, temperature=body.temperature or DEFAULT_TEMPERATURE
    )

    qwen_task = asyncio.create_task(call_ollama(QWEN_BASE_URL, qwen_payload))
    gptoss_task = asyncio.create_task(call_ollama(GPTOSS_BASE_URL, gptoss_payload))
    qwen_result, gptoss_result = await asyncio.gather(qwen_task, gptoss_task)
    # Fast-fail if either backend errored
    if qwen_result.get("error") or gptoss_result.get("error"):
        detail = {
            "qwen": {k: v for k, v in qwen_result.items() if k in ("error", "_base_url")},
            "gptoss": {k: v for k, v in gptoss_result.items() if k in ("error", "_base_url")},
        }
        return JSONResponse(status_code=502, content={"error": "backend_failed", "detail": detail})

    qwen_text = qwen_result.get("response", "")
    gptoss_text = gptoss_result.get("response", "")

    # 4) Optional brief debate (cross-critique)
    if ENABLE_DEBATE and MAX_DEBATE_TURNS > 0:
        critique_prompt = (
            "Critique the other model's answer. Identify mistakes, missing considerations, or improvements. "
            "Return a concise bullet list."
        )
        qwen_critique_msg = exec_messages + [ChatMessage(role="user", content=critique_prompt + f"\nOther answer:\n{gptoss_text}")]
        gptoss_critique_msg = exec_messages + [ChatMessage(role="user", content=critique_prompt + f"\nOther answer:\n{qwen_text}")]
        qwen_crit_payload = build_ollama_payload(qwen_critique_msg, QWEN_MODEL_ID, DEFAULT_NUM_CTX, body.temperature or DEFAULT_TEMPERATURE)
        gptoss_crit_payload = build_ollama_payload(gptoss_critique_msg, GPTOSS_MODEL_ID, DEFAULT_NUM_CTX, body.temperature or DEFAULT_TEMPERATURE)
        qcrit_task = asyncio.create_task(call_ollama(QWEN_BASE_URL, qwen_crit_payload))
        gcrit_task = asyncio.create_task(call_ollama(GPTOSS_BASE_URL, gptoss_crit_payload))
        qcrit_res, gcrit_res = await asyncio.gather(qcrit_task, gcrit_task)
        qcrit_text = qcrit_res.get("response", "")
        gcrit_text = gcrit_res.get("response", "")
        exec_messages = exec_messages + [
            ChatMessage(role="system", content="Cross-critique from Qwen:\n" + qcrit_text),
            ChatMessage(role="system", content="Cross-critique from GPT-OSS:\n" + gcrit_text),
        ]

    # 5) Final synthesis by Planner
    final_request = exec_messages + [
        ChatMessage(
            role="user",
            content=(
                "Produce the final, corrected answer, incorporating critiques and evidence. "
                "Be unambiguous, include runnable code when requested, and prefer specific citations to tool results."
            ),
        )
    ]

    planner_id = QWEN_MODEL_ID if PLANNER_MODEL.lower() == "qwen" else GPTOSS_MODEL_ID
    planner_base = QWEN_BASE_URL if PLANNER_MODEL.lower() == "qwen" else GPTOSS_BASE_URL
    synth_payload = build_ollama_payload(final_request, planner_id, DEFAULT_NUM_CTX, body.temperature or DEFAULT_TEMPERATURE)
    synth_result = await call_ollama(planner_base, synth_payload)
    final_text = synth_result.get("response", "") or qwen_text or gptoss_text

    if body.stream:
        async def _stream_final(text: str):
            head = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            yield f"data: {head}\n\n"
            content_chunk = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
            yield f"data: {content_chunk}\n\n"
            done = json.dumps({"id": "orc-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield f"data: {done}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream_final(final_text), media_type="text/event-stream")

    # Merge exact usages if available, else approximate
    usage = merge_usages([
        qwen_result.get("_usage"),
        gptoss_result.get("_usage"),
        synth_result.get("_usage"),
    ])
    if usage["total_tokens"] == 0:
        usage = estimate_usage(messages, final_text)

    response = ChatResponse(
        id="orc-1",
        model=f"committee:{QWEN_MODEL_ID}+{GPTOSS_MODEL_ID}",
        choices=[
            ChatChoice(index=0, finish_reason="stop", message={"role": "assistant", "content": final_text})
        ],
        usage=usage,
    )
    return JSONResponse(content=response.model_dump())


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/debug")
async def debug():
    # sanity checks to both backends
    try:
        qr = await call_ollama(QWEN_BASE_URL, {"model": QWEN_MODEL_ID, "prompt": "ping", "stream": False})
        gr = await call_ollama(GPTOSS_BASE_URL, {"model": GPTOSS_MODEL_ID, "prompt": "ping", "stream": False})
        return {"qwen_ok": "response" in qr and not qr.get("error"), "gptoss_ok": "response" in gr and not gr.get("error"), "qwen_detail": qr.get("error"), "gptoss_detail": gr.get("error")}
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": str(ex)})


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": f"committee:{QWEN_MODEL_ID}+{GPTOSS_MODEL_ID}", "object": "model"},
            {"id": QWEN_MODEL_ID, "object": "model"},
            {"id": GPTOSS_MODEL_ID, "object": "model"},
        ],
    }


