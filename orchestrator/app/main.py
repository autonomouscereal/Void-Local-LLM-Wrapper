from __future__ import annotations

import os
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

import httpx
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
async def call_ollama(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()


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

    response = ChatResponse(
        id="orc-1",
        model=f"committee:{QWEN_MODEL_ID}+{GPTOSS_MODEL_ID}",
        choices=[
            ChatChoice(index=0, finish_reason="stop", message={"role": "assistant", "content": final_text})
        ],
    )
    return JSONResponse(content=response.model_dump())


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


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


