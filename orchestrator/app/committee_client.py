"""
committee_client.py: Committee client for text generation (MINIMAL).
- NO logging
- NO emit_trace
- NO debate / critique / revision
- One Ollama call -> return
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import time

import httpx  # type: ignore

from .json_parser import JSONParser


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


def _render_prompt(messages: List[Dict[str, Any]]) -> str:
    rendered: List[str] = []
    for m in (messages or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "tool":
            tool_name = m.get("name") or "tool"
            rendered.append(f"tool[{tool_name}]: {m.get('content')}")
            continue

        content_val = m.get("content")
        if isinstance(content_val, list):
            text_parts: List[str] = []
            for part in content_val:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    txt = part.get("text")
                    if isinstance(txt, str):
                        text_parts.append(txt)
            rendered.append("\n".join(text_parts))
        else:
            rendered.append(str(content_val or ""))

    prompt = "\n\n".join(rendered)
    max_chars = int(DEFAULT_NUM_CTX) * 4
    if max_chars > 0 and len(prompt) > max_chars:
        prompt = prompt[-max_chars:]
    return prompt


def build_ollama_payload(messages: List[Dict[str, Any]], model: str, num_ctx: int, temperature: float) -> Dict[str, Any]:
    prompt = _render_prompt(messages)
    temp = float(temperature)
    if temp < 0.0:
        temp = 0.0
    if temp > 2.0:
        temp = 2.0
    ctx = int(num_ctx)
    if ctx < 256:
        ctx = 256

    return {
        "model": str(model),
        "prompt": prompt,
        "stream": False,
        "keep_alive": "24h",
        "options": {
            "num_ctx": int(ctx),
            "temperature": float(temp),
        },
    }


async def call_ollama(base_url: str, payload: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        r = await client.post(f"{(base_url or '').rstrip('/')}/api/generate", json=dict(payload))

    raw_text = r.text or ""
    status_code = int(r.status_code)

    parser = JSONParser()
    parsed_obj = parser.parse(raw_text or "{}", {})
    parsed = parsed_obj if isinstance(parsed_obj, dict) else {}

    if not (200 <= status_code < 300):
        return {
            "ok": False,
            "error": {
                "code": "ollama_http_error",
                "message": f"ollama returned HTTP {status_code}",
                "status": status_code,
                "base_url": str(base_url),
                "body": raw_text,
            },
        }

    response_str = parsed.get("response") if isinstance(parsed.get("response"), str) else ""
    response_str = _sanitize_mojibake_text(response_str or "")

    prompt_eval_val = parsed.get("prompt_eval_count")
    eval_count_val = parsed.get("eval_count")
    prompt_eval = int(prompt_eval_val) if isinstance(prompt_eval_val, (int, float)) else 0
    eval_count = int(eval_count_val) if isinstance(eval_count_val, (int, float)) else 0

    data: Dict[str, Any] = {
        "ok": True,
        "response": response_str,
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
        "dur_ms": int((time.monotonic() - t0) * 1000.0),
    }
    if prompt_eval or eval_count:
        data["_usage"] = {
            "prompt_tokens": int(prompt_eval),
            "completion_tokens": int(eval_count),
            "total_tokens": int(prompt_eval) + int(eval_count),
        }
    return data


async def committee_member_text(
    member_id: str,
    messages: List[Dict[str, Any]],
    *,
    trace_id: str,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    cfg = _participant_for(member_id=member_id)
    if not cfg:
        return {"ok": False, "error": {"code": "unknown_member", "message": f"unknown committee member: {member_id}"}}

    base = (cfg.get("base") or "").rstrip("/") or QWEN_BASE_URL
    model = cfg.get("model") or QWEN_MODEL_ID

    payload = build_ollama_payload(messages=messages or [], model=model, num_ctx=DEFAULT_NUM_CTX, temperature=temperature)
    res = await call_ollama(base_url=base, payload=payload, trace_id=trace_id)

    if isinstance(res, dict):
        res["_base_url"] = base
        res["_model"] = model
        res["_member"] = str(member_id or "")
    return res if isinstance(res, dict) else {"ok": False, "error": {"code": "bad_member_result", "message": str(res)}}


async def committee_ai_text(
    messages: List[Dict[str, Any]],
    *,
    trace_id: str,
    rounds: int | None = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    # MINIMAL: ignore rounds; single call to qwen (or COMMITTEE_PRIMARY_MEMBER)
    primary = os.getenv("COMMITTEE_PRIMARY_MEMBER", "qwen") or "qwen"
    res = await committee_member_text(primary, messages or [], trace_id=trace_id, temperature=float(temperature))

    text = ""
    if isinstance(res, dict) and res.get("ok") is not False:
        text = str(res.get("response") or "").strip()

    ok = bool(text)
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "ok": ok,
        "result": {"text": text, "primary": primary, "raw": res} if ok else None,
        "error": None if ok else {"code": "committee_no_answer", "message": "empty response", "raw": res},
    }


async def committee_jsonify(
    raw_text: str,
    expected_schema: Any,
    *,
    trace_id: str,
    rounds: int | None = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    env = await committee_ai_text(
        messages=[{"role": "user", "content": str(raw_text or "")}],
        trace_id=trace_id,
        rounds=rounds,
        temperature=float(temperature),
    )
    txt = ""
    if isinstance(env, dict) and isinstance(env.get("result"), dict):
        txt = str(env["result"].get("text") or "")
    parser = JSONParser()
    obj = parser.parse(txt or "{}", expected_schema)
    return obj if isinstance(obj, dict) else {}

def _schema_to_template(expected: Any) -> Any:
    if isinstance(expected, dict):
        return {k: _schema_to_template(v) for k, v in expected.items()}
    if isinstance(expected, list):
        if not expected:
            return []
        return [_schema_to_template(expected[0])]
    if isinstance(expected, type):
        if issubclass(expected, bool):
            return False
        if issubclass(expected, int):
            return 0
        if issubclass(expected, float):
            return 0.0
        if issubclass(expected, str):
            return ""
        if issubclass(expected, (list, tuple, set)):
            return []
        if issubclass(expected, dict):
            return {}
        return None
    return expected