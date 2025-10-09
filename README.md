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
   - `GPTOSS_MODEL_ID` default: `gpt-oss:20b`
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
   docker exec -it ollama_gptoss bash -lc "ollama pull gpt-oss:20b"
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
- Planner-Executor flow: Planner produces a plan + tool calls (semantic, not keyword-based). On refusals, the backend may force a single best-match tool with minimal args and re-synthesize, but it never overwrites the model’s content (only appends a Status/Tool Results section).
- MCP/tool-calling: Send a `tools` array (OpenAI-style JSON schema). The planner may propose `tool_calls` which the orchestrator executes when supported.
  - Built-in tools: `web_search` (SerpAPI), `run_python`, `write_file`, `read_file` (via Executor service).
  - Film tools (auto-exposed even if client doesn't declare them): `film_create`, `film_add_character`, `film_add_scene`, `film_compile`, `film_status`, `make_movie`.
  - Preferences: duration_seconds, resolution, fps, style, language, voice, quality, audio_enabled (default true), subtitles_enabled (default false), animation_enabled (default true)
  - MCP bridge: set `MCP_HTTP_BRIDGE_URL` to forward unknown tool calls prefixed with `mcp:` to your MCP HTTP server.
  - RAG tools (pgvector): `rag_index` (index `/workspace` into Postgres+pgvector), `rag_search` (retrieve top-k chunks).

Jobs API (ComfyUI long-running workflows)
-----------------------------------------
- POST `/jobs` with your ComfyUI `workflow` JSON to start a run. Response includes `job_id` and `prompt_id`.
- GET `/jobs/{job_id}` returns job state: `queued | running | succeeded | failed | cancelled`, recent checkpoints, and `result` when done.
- GET `/jobs/{job_id}/stream` streams SSE updates until completion. Optional `interval_ms` query param throttles updates.
- GET `/jobs` supports listing with optional `status`, `limit`, `offset`.
- POST `/jobs/{job_id}/cancel` attempts to interrupt job and marks as `cancelled`.

UI progress
-----------
- The chat UI renders full Markdown for assistant replies and appends a concise "### Tool Results" section (film_id, job_id(s), errors). If job_id(s) are present, the UI auto-subscribes to `/api/jobs/{id}/stream` and shows inline progress until done, then surfaces asset links.

Model keep-alive
----------------
- The orchestrator sends `options.keep_alive: "24h"` to Ollama so models stay resident and avoid reloading between calls. You can also set `OLLAMA_KEEP_ALIVE=24h` in the Ollama containers.

Persistence & RAG
-----------------
- Jobs persist to Postgres (`jobs`, `job_checkpoints` tables). On completion, if `JOBS_RAG_INDEX=true` (default), the job workflow and result are embedded into `rag_docs` for retrieval.

Film pipeline (LLM-driven)
--------------------------
- Built-in tools auto-exposed to planner: `make_movie`, `film_create`, `film_add_character`, `film_add_scene`, `film_status`, `film_compile`.
- A single prompt can create a film, define characters, spawn scene jobs, and compile automatically when scenes finish.
- Optional `N8N_WEBHOOK_URL` is called automatically once all scenes succeed; assembly result stored under `films.metadata`.
  - Local n8n is included in docker-compose and auto-imports the `Film Assemble` workflow at `/webhook/film-assemble`.
  - By default `N8N_WEBHOOK_URL` points to the internal n8n service: `http://n8n:5678/webhook/film-assemble`.
- PATCH endpoints for refinement:
  - PATCH `/films/{film_id}`: update film preferences (metadata)
  - PATCH `/characters/{character_id}`: update name/description/references (e.g., add more refs or set a character-specific voice)
  - PATCH `/scenes/{scene_id}`: update prompt/plan
- A film manifest JSON is written under `/uploads/film_<id>_manifest.json` on auto-compile for convenience.

Enhanced Streaming
------------------
- Chunked SSE streaming for `/v1/chat/completions` can be tuned via env vars:
  - `STREAM_CHUNK_SIZE_CHARS` (default 0 = single chunk)
  - `STREAM_CHUNK_INTERVAL_MS` (delay between chunks, default 50ms)
  - `JOBS_RAG_INDEX` (default `true`) to index job metadata into RAG on success

Executor service
----------------
- Mounted workspace: project root is available at `/workspace` in both orchestrator and executor.
- Endpoints:
  - POST `/run_python` { code }
  - POST `/write_file` { path, content }
  - POST `/read_file` { path }
- Environment knobs: `EXEC_TIMEOUT_SEC`, `EXEC_MEMORY_MB`, `ALLOW_SHELL` (optional `/run_shell` disabled by default).


