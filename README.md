Void Local LLM Committee (Qwen3-32B + GPT-OSS 20B)
===================================================

This repository brings up two Ollama services (Qwen and GPT-OSS), a sandboxed Executor, and a FastAPI orchestrator that exposes an OpenAI-compatible endpoint `/v1/chat/completions` which consults both models and synthesizes a final answer.

Prerequisites
-------------
- Windows 10/11 with WSL2 recommended, or native Linux
- NVIDIA GPU drivers and CUDA compatible GPUs (2x 24GB VRAM)
- Docker Desktop with NVIDIA Container Toolkit

Quick Start
-----------
1. Configure models via environment variables (optional):
   - `QWEN_MODEL_ID` default: `qwen2.5:32b-instruct-q4_K_M`
   - `GPTOSS_MODEL_ID` default: `gpt-oss:20b-q5_K_M`
   - `DEFAULT_NUM_CTX` default: `8192`
   - `DEFAULT_TEMPERATURE` default: `0.3`
   - `PLANNER_MODEL` default: `qwen` (values: `qwen` | `gptoss`)
   - `ENABLE_DEBATE` default: `true`
   - `MAX_DEBATE_TURNS` default: `1`
   - `ENABLE_WEBSEARCH` default: `true`
   - `ALLOW_TOOL_EXECUTION` default: `true`
   - `SERPAPI_API_KEY` set to enable web search

2. Start services:
   ```bash
   # Ensure Docker Desktop is running
   docker compose up -d --build
   ```

3. Pull models in each Ollama container (first time only):
   ```bash
   docker exec -it ollama_qwen bash -lc "ollama pull $QWEN_MODEL_ID"
   docker exec -it ollama_gptoss bash -lc "ollama pull $GPTOSS_MODEL_ID"
   ```

   If environment variables are not set, replace with explicit tags, e.g.:
   ```bash
   docker exec -it ollama_qwen bash -lc "ollama pull qwen2.5:32b-instruct-q4_K_M"
   docker exec -it ollama_gptoss bash -lc "ollama pull gpt-oss:20b-q5_K_M"
   ```

4. Test orchestrator:
   ```bash
   curl http://localhost:8000/healthz
   ```

OpenAI-Compatible Usage
-----------------------
Point your client (e.g., Void or OpenAI SDK) to `http://localhost:8000` and call `/v1/chat/completions`.
- Tool-calling semantics (default: client executes):
  - `AUTO_EXECUTE_TOOLS=false` (default) → endpoint returns `tool_calls` for Void to execute; send results back as `role: "tool"` messages.
  - `AUTO_EXECUTE_TOOLS=true` → backend executes supported tools automatically.


Example request:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Draft a Python function to quicksort a list."}
        ]
      }'
```

Notes
-----
- GPU pinning: `ollama_qwen` uses GPU 0 (port 11434), `ollama_gptoss` uses GPU 1 (mapped to 11435).
- Adjust quantization by changing `QWEN_MODEL_ID` and `GPTOSS_MODEL_ID` to quantized tags (e.g., `-q4_K_M`, `-q5_K_M`, `-q8_0`).
- Increase `DEFAULT_NUM_CTX` if your quantization and VRAM permit.
- Planner-Executor flow: Planner produces a plan + tool calls, tools execute (web_search via SerpAPI), both models respond, cross-critique, planner synthesizes final.
- MCP/tool-calling: Send a `tools` array (OpenAI-style JSON schema). The planner may propose `tool_calls` which the orchestrator executes when supported.
  - Built-in tools: `web_search` (SerpAPI), `run_python`, `write_file`, `read_file` (via Executor service).
  - MCP bridge: set `MCP_HTTP_BRIDGE_URL` to forward unknown tool calls prefixed with `mcp:` to your MCP HTTP server.
  - RAG tools (pgvector): `rag_index` (index `/workspace` into Postgres+pgvector), `rag_search` (retrieve top-k chunks).

Executor service
----------------
- Mounted workspace: project root is available at `/workspace` in both orchestrator and executor.
- Endpoints:
  - POST `/run_python` { code }
  - POST `/write_file` { path, content }
  - POST `/read_file` { path }
- Environment knobs: `EXEC_TIMEOUT_SEC`, `EXEC_MEMORY_MB`, `ALLOW_SHELL` (optional `/run_shell` disabled by default).


