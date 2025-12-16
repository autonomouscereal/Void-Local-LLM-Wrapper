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

import requests  # type: ignore

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


def build_ollama_payload(messages: List[Dict[str, Any]], model: str, num_ctx: int, temperature: float):
    prompt = (messages)

    return {
        "model": str(model),
        "prompt": prompt,
        "stream": False,
        "keep_alive": "24h",
        "options": {
            "num_ctx": int(8000),
        },
    }


async def call_ollama(base_url: str, payload: Dict[str, Any], trace_id: str):
    url = f"{(base_url or '').rstrip('/')}/api/generate"
    
    with requests.Session() as s:
        # Mirror httpx trust_env=False: ignore proxy env vars (prevents odd corporate proxy issues).
        r = s.post(url, json=payload)
    
    status_code = int(getattr(r, "status_code", 0) or 0)
    parsed_any: Any = None
    raw_text = ""
    parsed_any = r.json()
    
    # Normalize to dict-like payload (Ollama should return JSON, but we fail-soft).
    parsed: Dict[str, Any] = parsed_any if isinstance(parsed_any, dict) else {}
    if not parsed and raw_text:
        parser = JSONParser()
        parsed_obj = parser.parse(raw_text or "{}", {})
        parsed = parsed_obj if isinstance(parsed_obj, dict) else {}

    if not (200 <= status_code < 300):
        body: Any = parsed_any if parsed_any is not None else raw_text
        return {
            "ok": False,
            "error": {
                "code": "ollama_http_error",
                "message": f"ollama returned HTTP {status_code}",
                "status": status_code,
                "base_url": str(base_url),
                "body": body,
            },
        }

    response_str = parsed.get("response") if isinstance(parsed.get("response"), str) else ""
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
    if not cfg:
        return {"ok": False, "error": {"code": "unknown_member", "message": f"unknown committee member: {member_id}"}}

    base = (cfg.get("base") or "").rstrip("/") or QWEN_BASE_URL
    model = cfg.get("model") or QWEN_MODEL_ID

    payload = build_ollama_payload(messages=messages, model=model, num_ctx=DEFAULT_NUM_CTX)
    res = await call_ollama(base_url=base, payload=payload, trace_id=trace_id)

    if isinstance(res, dict):
        res["_base_url"] = base
        res["_model"] = model
        res["_member"] = str(member_id or "")
    return res


async def committee_ai_text(messages: List[Dict[str, Any]], trace_id: str, rounds: int | None = None, temperature: float = 0.3):
    member_id = 'qwen'
    res = await committee_member_text(member_id, messages or [], trace_id=trace_id)

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


async def committee_jsonify(raw_text: str, expected_schema: Any, trace_id: str, rounds: int | None = None, temperature: float = 0.0):
    return  {}