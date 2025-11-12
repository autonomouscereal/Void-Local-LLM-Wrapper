from __future__ import annotations

import os
import asyncio
import json
from typing import Dict, Any

import httpx  # type: ignore

from ..json_parser import JSONParser
from ..pipeline.compression_orchestrator import co_pack, frames_to_string


PLAN_SCHEMA: Dict[str, Any] = {
    "request_id": str,
    "plan": [
        {
            "id": str,
            "tool": str,
            "inputs": dict,
            "needs": [str],
            "provides": dict,
        }
    ],
}


SYSTEM_IMAGE = (
    "You are ImageOps. Output ONLY JSON per the schema. "
    "If an image is requested, include image.dispatch and sub-steps for pose/edge/depth/style locks when relevant."
)
SYSTEM_VIDEO = (
    "You are VideoOps. Output ONLY JSON per the schema. "
    "If any visual content is requested, include t2v/i2v plus stabilize and video.upscale where relevant."
)
SYSTEM_AUDIO = (
    "You are AudioOps. Output ONLY JSON per the schema. "
    "If any audio/music/voice is implied, include music generate and optional vocal + master/mix steps."
)


async def _call_llm_json(prompt: str) -> Dict[str, Any]:
    base = os.getenv("QWEN_BASE_URL", "http://ollama_qwen:11434").rstrip("/")
    model = os.getenv("QWEN_MODEL_ID", "qwen2.5:14b").strip() or "qwen2.5:14b"
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient() as client:
        r = await client.post(base + "/api/generate", json=payload)
        r.raise_for_status()
        txt = (JSONParser().parse(r.text, {"response": str}) or {}).get("response") or "{}"
        try:
            return JSONParser().parse(txt, PLAN_SCHEMA)
        except Exception:
            return {"request_id": "", "plan": []}


async def _plan_with_role(system: str, user_text: str, request_id: str) -> Dict[str, Any]:
    # Build CO frames for planner/committee via internal pack (prompt-only, no enforcement)
    co_env = {
        "schema_version": 1,
        "trace_id": request_id,
        "call_kind": "planner",
        "model_caps": {"num_ctx": 8192},
        "user_turn": {"role": "user", "content": user_text},
        "history": [],
        "attachments": [],
        "tool_memory": [],
        "rag_hints": [],
        "roe_incoming_instructions": [],
        "subject_canon": {},
        "percent_budget": {"icw_pct": [65, 70], "tools_pct": [18, 20], "roe_pct": [5, 10], "misc_pct": [3, 5], "buffer_pct": 5},
        "sweep_plan": ["0-90", "30-120", "60-150+wrap"],
    }
    co_out = co_pack(co_env)
    frames_text = frames_to_string(co_out.get("frames") or [])
    prompt = (
        frames_text
        + "\n\n"
        + f"Return JSON exactly matching this schema: {json.dumps(PLAN_SCHEMA, ensure_ascii=False)}\n"
        + f"User request: {user_text}\n"
        + f"Your role: {system}\n"
        + "Include ALL relevant steps for your modality; do not omit sub-steps.\n"
        + "### [PLANNER / SYSTEM]\n"
        + "Plan minimal steps, prefer tools over text; for image.dispatch integrate subject canon when applicable and key args; merge to single plan and note uncertainties.\n"
        + "### [FINALITY / SYSTEM]\n"
        + "Do not output assistant content until committee consensus; one final answer only."
    )
    out = await _call_llm_json(prompt)
    out["request_id"] = request_id
    return out


async def _plan_with_retry(system: str, user_text: str, request_id: str, attempts: int = 2) -> Dict[str, Any]:
    for i in range(attempts):
        try:
            return await _plan_with_role(system, user_text, request_id)
        except Exception:
            if i == attempts - 1:
                return {"request_id": request_id, "plan": []}


async def make_full_plan(user_text: str) -> Dict[str, Any]:
    import uuid as _uuid
    rid = str(_uuid.uuid4())
    img_t = asyncio.create_task(_plan_with_retry(SYSTEM_IMAGE, user_text, rid))
    vid_t = asyncio.create_task(_plan_with_retry(SYSTEM_VIDEO, user_text, rid))
    aud_t = asyncio.create_task(_plan_with_retry(SYSTEM_AUDIO, user_text, rid))
    parts = await asyncio.gather(img_t, vid_t, aud_t, return_exceptions=True)
    merged = {"request_id": rid, "plan": []}
    seen = set()
    for p in parts:
        if not isinstance(p, dict):
            continue
        for step in p.get("plan", []):
            key = (step.get("tool"), json.dumps(step.get("inputs", {}), ensure_ascii=False, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            merged["plan"].append(step)
    # Post-process: unify all video intents under film2.run (no bare video.*)
    has_video = any(isinstance(s, dict) and isinstance(s.get("tool"), str) and s.get("tool", "").startswith("video.") for s in (merged.get("plan") or []))
    if has_video:
        merged["plan"] = [s for s in merged.get("plan") or [] if not (isinstance(s, dict) and isinstance(s.get("tool"), str) and s.get("tool", "").startswith("video."))]
        merged["plan"].append({"id": "film-1", "tool": "film2.run", "inputs": {"prompt": user_text}, "needs": [], "provides": {}})
    if not (merged.get("plan") or []):
        merged["plan"] = [{"id": "img-1", "tool": "image.dispatch", "inputs": {"prompt": user_text}, "needs": [], "provides": {}}]
    return merged


