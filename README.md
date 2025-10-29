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
   - `QWEN_MODEL_ID` default: `qwen3:32b-instruct-q4_K_M`
   - `GPTOSS_MODEL_ID` default: `gpt-oss:20b`
   - `DEFAULT_NUM_CTX` default: `8192`
   - `DEFAULT_TEMPERATURE` default: `0.3`
   - `PLANNER_MODEL` default: `qwen3` (values: `qwen3` | `qwen` | `gptoss`)
   - `ENABLE_DEBATE` default: `true`
   - `MAX_DEBATE_TURNS` default: `1`
   - `ENABLE_WEBSEARCH` default: `true`
   - `AUTO_EXECUTE_TOOLS` default: `true`
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
   docker exec -it ollama_qwen bash -lc "ollama pull qwen3:32b-instruct-q4_K_M"
   docker exec -it ollama_gptoss bash -lc "ollama pull gpt-oss:20b"
   ```

4. Test orchestrator:
   ```bash
   curl http://localhost:8000/healthz
   ```

OpenAI-Compatible Usage
-----------------------
Point your client (e.g., Void or OpenAI SDK) to `http://localhost:8000` and call `/v1/chat/completions`.
- Tool-calling semantics (server executes by default):
  - `AUTO_EXECUTE_TOOLS=true` (default) → backend executes supported tools automatically.
  - Set `AUTO_EXECUTE_TOOLS=false` only if you want the client to receive `tool_calls` and execute tools itself.


Single‑prompt film example (server orchestrates tools automatically):
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "wrapper-film-orchestrator",
        "messages": [
          {"role": "user", "content": "Make a 3-minute 4K 60fps film about monarch butterflies with interpolation and motion smoothing. Title: Butterflies."}
        ],
        "tool_choice": "auto",
        "stream": true
      }'
```

Notes
-----
- GPU pinning: `ollama_qwen` uses GPU 0 (port 11434), `ollama_gptoss` uses GPU 1 (mapped to 11435).
- Adjust quantization by changing `QWEN_MODEL_ID` and `GPTOSS_MODEL_ID` to quantized tags (e.g., `-q4_K_M`, `-q5_K_M`, `-q8_0`).
- Increase `DEFAULT_NUM_CTX` if your quantization and VRAM permit.
- Planner-Executor flow: Planner produces a plan + tool calls (semantic, not keyword-based). On refusals, the backend may force a single best-match tool with minimal args and re-synthesize, but it never overwrites the model’s content (only appends a Status/Tool Results section).
- MCP/tool-calling: Send a `tools` array (OpenAI-style JSON schema). The planner may propose `tool_calls` which the orchestrator executes when supported.
  - Built-in tools: `web_search` (SerpAPI), `metasearch.fuse` (multi‑engine rank fusion), `run_python`, `write_file`, `read_file` (via Executor service).
  - Film tools: Film‑1 removed. A single internal Film‑2 orchestrator is used (`film.run`).
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

Film‑2 pipeline (LLM-driven)
----------------------------
- Server uses a single internal tool `film.run` to orchestrate: `plan → breakdown → storyboard → animatic → final → post → qc → export`.
- Per‑stage manifests are persisted (DB authoritative; JSON optional): `plan/scenes/characters`, `shots (DSL + seeds)`, per‑shot `storyboard/animatic/final/nodes`, project‑level `edl/qc/export`.
- n8n is optional. If configured, it may be invoked at export; otherwise local assembler is used.
- PATCH endpoints remain for refinement:
  - PATCH `/films/{film_id}`: update film preferences (metadata)
  - PATCH `/characters/{character_id}`: update name/description/references
  - PATCH `/scenes/{scene_id}`: update prompt/plan

Capabilities and Health
-----------------------
- `GET /capabilities.json` advertises endpoints, versions, tool allowlist, and config hash for IDE discovery.
- `GET /healthz` includes: `{ ok, openai_compat, teacher_enabled, icw_enabled, film_enabled, ablation_enabled }`.

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

Scaling & Multi‑GPU ComfyUI
---------------------------
- Current default: single ComfyUI instance (CPU‑mode is fine for validation).
- To scale without code changes:
  1) Add more ComfyUI services (e.g., `comfyui-1`, `comfyui-2`) using the same image and shared volumes.
     - For GPU pinning, set per‑service:
       - `runtime: nvidia`
       - `environment: [NVIDIA_VISIBLE_DEVICES=<gpu_id>, NVIDIA_DRIVER_CAPABILITIES=compute,utility]`
  2) Set the orchestrator envs:
     - `COMFYUI_API_URLS` to a comma‑separated list of instance URLs (e.g., `http://comfyui:8188,http://comfyui-1:8188`)
     - `SCENE_SUBMIT_CONCURRENCY` to your desired parallelism (e.g., `4`)
  3) Leave GPUs 0,1 reserved for `ollama_qwen` and `ollama_gptoss`. Use other GPU IDs for ComfyUI instances.
- Behavior:
  - The orchestrator picks the least‑loaded ComfyUI instance for each submit and pins the job’s `prompt_id` to that instance for history polling.
  - If one instance fails, it falls back to others automatically.
- Example `.env` for 2 ComfyUI GPUs (IDs 2 and 3):
  ```env
  COMFYUI_API_URLS=http://comfyui:8188,http://comfyui-1:8188
  SCENE_SUBMIT_CONCURRENCY=4
  # In compose override, set comfyui-1 with NVIDIA_VISIBLE_DEVICES=3 (comfyui uses 2)
  ```

Inline Safety Notes
-------------------
- Orchestrator warns against SQLAlchemy and uses asyncpg exclusively for DB access.
- JSON parsing uses a hardened parser to survive malformed LLM outputs.
- Film consistency: character face embeddings are computed once and propagated to each scene; seeds are deterministic for every tool call and stage.

Model defaults
--------------
- Planner/Executors default to Qwen‑3 for the primary model (`QWEN_MODEL_ID=qwen3:32b-instruct-q4_K_M`).
- Ablation judge/compressor is routed to Qwen‑3 (via `ABLCODER_URL`).
- Jobs persist as failed even on submit/network errors so the UI never shows an empty queue during errors.


