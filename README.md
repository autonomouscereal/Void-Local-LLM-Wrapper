Void Orchestrator — Multimodal, Deterministic, Max‑Quality by Default
====================================================================

Overview
--------
OpenAI‑compatible orchestrator with deterministic routing, capless WindowedSolver (CONT/HALT), JSON‑only envelopes, RAG hygiene, ablation, Film‑2, Images/TTS/Music, and full traceability for distillation.

Quick Start (copy/paste)
------------------------
- Start everything with your env file:
  ```bash
  docker compose --env-file env.example.txt up -d --build
  ```
- Health check:
  ```bash
  curl http://localhost:8000/healthz
  ```
- Chat (OpenAI‑compatible): POST `http://localhost:8000/v1/chat/completions`

What’s Included (always ON)
--------------------------
- WindowedSolver: sliding window + CAC; CONT/HALT; checkpoints; seed registry; no inline caps.
- Deterministic routing: Film (`film.run`), Images (`image.dispatch`), TTS (`tts.speak`), Music (song/instrumental/infinite composer), RAG, Research, Code Super‑Loop.
- Images: super‑gen, edit, upscale; face/pose/style locks; hands/artifact fixes; inline assets.
- Music/TTS: full songs (YuE) + instrumentals (MusicGen) + duration‑locked (SAO); infinite composer with bar‑aligned seams, seam QA/auto‑repair, mastering; XTTS voice‑lock; stems; manifests.
- Film‑2: clarifications → storyboard/animatic/final; shot‑level locks; QA/re‑render; upscale/interp; EDL/export.
- Research/RAG: keyless multi‑engine `metasearch.fuse`, RAG TTL/dedup/newest‑first; evidence binding.
- Ablation: facts extraction + `facts.jsonl` export; attached to envelopes.
- Traceability: envelopes, manifests (hashes/seeds/params/GPU), job ledger shards, dataset exports.

Essential Commands
------------------
- Bring up services with a specific env file:
  ```bash
  docker compose --env-file env.example.txt up -d --build
  ```
- Tail orchestrator logs:
  ```bash
  docker logs -f orchestrator
  ```
- UI (Chat): open the Chat UI (service) and use enter‑to‑send; assets render inline.

Key Endpoints
-------------
- `GET /healthz` — health
- `GET /capabilities.json` — live capabilities
- `POST /v1/chat/completions` — OpenAI‑compatible chat (tools auto‑executed)
- `POST /tool.run` — direct tool execution
- Jobs (long ComfyUI flows): `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/stream`, `POST /jobs/{id}/cancel`

Determinism & Memory
--------------------
- Seeds: deterministic seeds per tool/model; stamped into envelopes and manifests.
- Memory: multimodal artifact index (JSONL); cross‑conversation recall.

GPU Notes
---------
- CUDA GPUs required; NVIDIA toolkit via Docker; CDI device exposure. Heavy models prefer RTX 8000; P40 pool used for concurrency. OOM auto‑mitigated via window sizing/retry (continuity preserved).

License
-------
Proprietary / Internal use.


