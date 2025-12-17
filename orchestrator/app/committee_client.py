"""
committee_client.py: Committee client for text generation (MINIMAL).
- NO logging
- NO emit_trace
- NO debate / critique / revision
- One Ollama call -> return
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import os
import time

import httpx  # type: ignore
import subprocess
import json

from .json_parser import JSONParser

logger = logging.getLogger(__name__)

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11435")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen3:30b-a3b-instruct-2507-q4_K_M")

GLM_OLLAMA_BASE_URL = os.getenv("GLM_OLLAMA_BASE_URL", "http://localhost:11433")
GLM_MODEL_ID = os.getenv("GLM_MODEL_ID", "glm4:9b")

DEEPSEEK_CODER_OLLAMA_BASE_URL = os.getenv("DEEPSEEK_CODER_OLLAMA_BASE_URL", "http://localhost:11436")
DEEPSEEK_CODER_MODEL_ID = os.getenv("DEEPSEEK_CODER_MODEL_ID", "deepseek-coder-v2:lite")

_DEFAULT_NUM_CTX_RAW = os.getenv("DEFAULT_NUM_CTX", "8192")
try:
    DEFAULT_NUM_CTX = int(str(_DEFAULT_NUM_CTX_RAW).strip() or "8192")
except Exception:
    DEFAULT_NUM_CTX = 8192

_COMMITTEE_ROUNDS_RAW = os.getenv("COMMITTEE_ROUNDS", "1")
try:
    DEFAULT_COMMITTEE_ROUNDS = int(str(_COMMITTEE_ROUNDS_RAW).strip() or "1")
except Exception:
    DEFAULT_COMMITTEE_ROUNDS = 1

COMMITTEE_MODEL_ID = os.getenv("COMMITTEE_MODEL_ID") or f"committee:{QWEN_MODEL_ID}+{GLM_MODEL_ID}+{DEEPSEEK_CODER_MODEL_ID}"

PARTICIPANT_MODELS: Dict[str, Dict[str, str]] = {
    "qwen": {"base": QWEN_BASE_URL, "model": QWEN_MODEL_ID},
    "glm": {"base": GLM_OLLAMA_BASE_URL, "model": GLM_MODEL_ID},
    "deepseek": {"base": DEEPSEEK_CODER_OLLAMA_BASE_URL, "model": DEEPSEEK_CODER_MODEL_ID},
}


def _sanitize_mojibake_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    replacements = (
        ("Ã¢Â\x80Â\x99", "'"),
        ("Ã¢ÂÂ", "'"),
        ("â\x80\x99", "'"),
        ("â€™", "'"),
        ("â", "'"),
    )
    out = text
    for bad, good in replacements:
        if bad in out:
            out = out.replace(bad, good)
    return out


def _participant_for(member_id: str) -> Optional[Dict[str, str]]:
    cfg = PARTICIPANT_MODELS.get(str(member_id or "").strip())
    return cfg if isinstance(cfg, dict) else None


def build_ollama_payload(messages: List[Dict[str, Any]], model: str, num_ctx: int, temperature: float = 0.3):
    logger.info(f"build_ollama_payload: messages={messages}, model={model}, num_ctx={num_ctx}, temperature={temperature}")
    
    payload = {
        "model": str(model),
        "messages": messages,
        "stream": False,
        "keep_alive": "24h",
        "options": {
            "num_ctx": int(num_ctx),
        },
    }

    logger.info(f"build_ollama_payload: payload={payload}")

    return payload


async def call_ollama(base_url: str, payload: Dict[str, Any], trace_id: str):

    logger.info(f"call_ollama: base_url={base_url}, payload={payload}, trace_id={trace_id}")
    url = f"{(base_url or '').rstrip('/')}/api/chat"
    
    parsed = {}

    # Fully blocking "POST" using curl to bypass Python HTTP stack entirely.
    # (This still blocks the request exactly as required.)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    p = subprocess.run(
        ["curl", "-sS", "-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@-", url],
        input=body,
        capture_output=True,
    )
    raw_text = (p.stdout or b"").decode("utf-8", errors="replace")

    logger.info(f"ollama response: {raw_text}")
    parser = JSONParser()
    parsed_obj = parser.parse(raw_text, {})
    parsed = parsed_obj if isinstance(parsed_obj, dict) else {}

    # Ollama has two common shapes:
    # - /api/generate -> {"response": "..."}
    # - /api/chat     -> {"message": {"content": "..."}, ...}
    response_str = ""
    if isinstance(parsed.get("response"), str):
        response_str = str(parsed.get("response") or "")
    else:
        msg = parsed.get("message") if isinstance(parsed.get("message"), dict) else {}
        if isinstance(msg.get("content"), str):
            response_str = str(msg.get("content") or "")
    response_str = _sanitize_mojibake_text(response_str or "")

    prompt_eval_val = parsed.get("prompt_eval_count")
    eval_count_val = parsed.get("eval_count")
    prompt_eval = int(prompt_eval_val) if isinstance(prompt_eval_val, (int, float)) else 0
    eval_count = int(eval_count_val) if isinstance(eval_count_val, (int, float)) else 0

    logger.info(f"ollama response: {response_str}")

    out: Dict[str, Any] = {
        "ok": True,
        "response": response_str,
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
    }
    return out


async def committee_member_text(member_id: str, messages: List[Dict[str, Any]], trace_id: str, temperature: float = 0.3):
    cfg = _participant_for(member_id=member_id)

    logger.info(f"committee_ai_text: messages={messages}")

    base = (cfg.get("base") or "").rstrip("/") or QWEN_BASE_URL
    model = cfg.get("model") or QWEN_MODEL_ID

    payload = build_ollama_payload(messages=messages, model=model, num_ctx=DEFAULT_NUM_CTX)
    res = await call_ollama(base_url=base, payload=payload, trace_id=trace_id)

    if isinstance(res, dict):
        res["_base_url"] = base
        res["_model"] = model
        res["_member"] = str(member_id or "")
    return res


async def committee_ai_text(messages , trace_id: str, rounds: int = None, temperature: float = 0.3):
    member_id = 'qwen'
    logger.info(f"committee_ai_text: messages={messages}")
    res = await committee_member_text(member_id, messages, trace_id=trace_id)

    text = ""
    if isinstance(res, dict) and res.get("ok") is not False:
        text = str(res.get("response") or "").strip()

    ok = bool(text)
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "ok": ok,
        "result": {"text": text, "committee_member": member_id, "raw": res} if ok else None,
        "error": None if ok else {"code": "committee_no_answer", "message": "empty response", "raw": res},
    }


async def committee_jsonify(raw_text: str, expected_schema: Any, trace_id: str, rounds: int = None, temperature: float = 0.3):
    return  {}