from __future__ import annotations

from typing import Any, Dict, List

import os
import json

import httpx  # type: ignore

from .json_parser import JSONParser
from .pipeline.compression_orchestrator import co_pack, frames_to_messages
from .trace_utils import emit_trace


# Committee model routing and context configuration.
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11435")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen3:30b-a3b-instruct-2507-q4_K_M")
GLM_OLLAMA_BASE_URL = os.getenv("GLM_OLLAMA_BASE_URL", "http://localhost:11433")
GLM_MODEL_ID = os.getenv("GLM_MODEL_ID", "glm4:9b")
DEEPSEEK_CODER_OLLAMA_BASE_URL = os.getenv("DEEPSEEK_CODER_OLLAMA_BASE_URL", "http://localhost:11436")
DEEPSEEK_CODER_MODEL_ID = os.getenv("DEEPSEEK_CODER_MODEL_ID", "deepseek-coder-v2:lite")
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "8192"))
COMMITTEE_MODEL_ID = os.getenv("COMMITTEE_MODEL_ID") or f"committee:{QWEN_MODEL_ID}+{GLM_MODEL_ID}+{DEEPSEEK_CODER_MODEL_ID}"
DEFAULT_COMMITTEE_ROUNDS = max(1, int(os.getenv("COMMITTEE_ROUNDS", "1") or "1"))

PARTICIPANT_MODELS: Dict[str, Dict[str, str]] = {
    "qwen": {
        "base": QWEN_BASE_URL,
        "model": QWEN_MODEL_ID,
    },
    "glm": {
        "base": GLM_OLLAMA_BASE_URL,
        "model": GLM_MODEL_ID,
    },
    "deepseek": {
        "base": DEEPSEEK_CODER_OLLAMA_BASE_URL,
        "model": DEEPSEEK_CODER_MODEL_ID,
    },
}
COMMITTEE_PARTICIPANTS = [
    {"id": name, "base": cfg["base"], "model": cfg["model"]}
    for name, cfg in PARTICIPANT_MODELS.items()
]

# Local state dir for committee traces (mirrors main.STATE_DIR logic).
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
STATE_DIR = os.path.join(UPLOAD_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)


async def call_ollama(base_url: str, payload: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    """
    Low-level Ollama generate call used by committee debate.

    This function is owned by the committee layer so that all committee-related
    backend calls are centralized here.
    """
    trace_key = str(trace_id or "committee")
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        # Trace request (without full prompt content to avoid huge payloads).
        emit_trace(
            STATE_DIR,
            trace_key,
            "committee.ollama.request",
            {
                "trace_id": trace_key,
                "base_url": base_url,
                "model": payload.get("model"),
                "num_ctx": payload.get("num_ctx"),
                "temperature": payload.get("temperature"),
            },
        )
        ppayload = dict(payload)
        resp = await client.post(f"{base_url}/api/generate", json=ppayload)
        parser = JSONParser()
        data = parser.parse(resp.text or "", {"response": str, "prompt_eval_count": int, "eval_count": int})
        usage: Dict[str, int] | None = None
        if isinstance(data, dict) and ("prompt_eval_count" in data or "eval_count" in data):
            usage = {
                "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
                "completion_tokens": int(data.get("eval_count", 0) or 0),
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            data["_usage"] = usage
        # Prepare a short preview of the model response for logging/tracing.
        resp_text = ""
        if isinstance(data, dict):
            raw_resp = data.get("response")
            if isinstance(raw_resp, str):
                resp_text = raw_resp.strip()
        preview = resp_text[:512] if resp_text else ""
        emit_trace(
            STATE_DIR,
            trace_key,
            "committee.ollama.response",
            {
                "trace_id": trace_key,
                "status_code": int(getattr(resp, "status_code", 0) or 0),
                "usage": usage or {},
                "response_preview": preview,
            },
        )
        return data


def build_ollama_payload(messages: List[Dict[str, Any]], model: str, num_ctx: int, temperature: float) -> Dict[str, Any]:
    rendered: List[str] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            tool_name = m.get("name") or "tool"
            rendered.append(f"tool[{tool_name}]: {m.get('content')}")
        else:
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
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "num_ctx": int(num_ctx),
        "temperature": float(temperature),
    }


async def committee_ai_text(
    messages: List[Dict[str, Any]],
    trace_id: str,
    rounds: int | None = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Core committee primitive: all participants see the same base messages,
    then iteratively see each other's answers and refine their own.

    The returned envelope always contains:
    - result.text: final synthesized answer (used as the primary output)
    - result.members: light per-member summaries
    - result.qwen / result.glm / result.deepseek (when present): the latest
      raw backend envelopes from each participant, including `_usage`
      and `_base_url` for compatibility with legacy callers.
    - result.synth: the envelope corresponding to the member whose
      answer was selected as `result.text` (or a minimal fallback).
    """
    effective_rounds = max(1, int(rounds if rounds is not None else DEFAULT_COMMITTEE_ROUNDS))
    answers: Dict[str, str] = {}
    # Full per-member envelopes as returned from call_ollama
    member_results: Dict[str, Dict[str, Any]] = {}
    emit_trace(
        STATE_DIR,
        str(trace_id or "committee"),
        "committee.start",
        {
            "trace_id": str(trace_id or "committee"),
        "rounds": int(effective_rounds),
            "temperature": float(temperature),
            "participants": [p.get("id") or "member" for p in COMMITTEE_PARTICIPANTS],
            "messages": messages or [],
        },
    )
    for r in range(effective_rounds):
        for p in COMMITTEE_PARTICIPANTS:
            member = p.get("id") or "member"
            base = (p.get("base") or "").rstrip("/") or QWEN_BASE_URL
            model = p.get("model") or QWEN_MODEL_ID
            member_msgs: List[Dict[str, Any]] = list(messages or [])
            ctx_lines: List[str] = []
            if r == 0 and not answers:
                ctx_lines.append(
                    f"You are committee member {member}. Provide your best answer to the user."
                )
            else:
                ctx_lines.append(
                    f"You are committee member {member}. This is debate round {r+1}."
                )
                for other_id, ans in answers.items():
                    if other_id == member or not isinstance(ans, str) or not ans.strip():
                        continue
                    ctx_lines.append(f"Answer from {other_id}:\n{ans.strip()[:800]}")
                prior = answers.get(member)
                if isinstance(prior, str) and prior.strip():
                    ctx_lines.append(f"Your previous answer was:\n{prior.strip()[:800]}")
                ctx_lines.append(
                    "Compare all answers, keep what is correct, fix what is wrong, "
                    "and produce your best updated answer. Respond with ONLY your full answer."
                )
            member_msgs.append({"role": "system", "content": "\n\n".join(ctx_lines)})
            emit_trace(
                STATE_DIR,
                str(trace_id or "committee"),
                "committee.member_round",
                {
                    "trace_id": str(trace_id or "committee"),
                    "round": int(r + 1),
                    "member": member,
                    "model": model,
                    "base": base,
                    "messages": member_msgs,
                },
            )
            payload = build_ollama_payload(member_msgs, model, DEFAULT_NUM_CTX, temperature)
            res = await call_ollama(base, payload, trace_id=trace_id)
            if isinstance(res, dict):
                # Annotate with backend base URL for downstream diagnostics.
                res["_base_url"] = base
                member_results[member] = res
                txt = (res.get("response") or "").strip()
            else:
                member_results[member] = {}
                txt = ""
            answers[member] = txt

    final_text = ""
    synth_member_id: str | None = None
    member_summaries: List[Dict[str, Any]] = []
    for p in COMMITTEE_PARTICIPANTS:
        mid = p.get("id") or "member"
        ans = answers.get(mid) or ""
        if not final_text and isinstance(ans, str) and ans.strip():
            final_text = ans.strip()
            synth_member_id = mid
        member_summaries.append({"member": mid, "answer": ans})

    # Build structured result payload with per-member views + synth envelope.
    result_payload: Dict[str, Any] = {
        "text": final_text,
        "members": member_summaries,
    }
    # Attach per-backend envelopes when present so callers can inspect them directly.
    for key in ("qwen", "glm", "deepseek"):
        if key in member_results:
            result_payload[key] = member_results.get(key) or {}
    # Derive a synth envelope from the member whose answer we selected as final_text.
    synth_result: Dict[str, Any] = {}
    if synth_member_id and synth_member_id in member_results:
        synth_src = member_results.get(synth_member_id) or {}
        # Shallow copy so callers can mutate without touching internal cache.
        synth_result = dict(synth_src)
    else:
        # Minimal fallback if no member produced a non-empty answer.
        synth_result = {"response": final_text}
    result_payload["synth"] = synth_result
    ok = bool(final_text)
    emit_trace(
        STATE_DIR,
        str(trace_id or "committee"),
        "committee.finish",
        {
            "trace_id": str(trace_id or "committee"),
            "ok": ok,
            "final_text": final_text,
            "members": member_summaries,
        },
    )
    return {
        "schema_version": 1,
        "trace_id": trace_id or "committee",
        "ok": ok,
        "result": result_payload if ok else None,
        "error": None if ok else {
            "code": "committee_no_answer",
            "message": "Committee did not produce a non-empty answer",
        },
    }


async def committee_jsonify(
    raw_text: str,
    expected_schema: Any,
    *,
    trace_id: str,
    rounds: int | None = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Use the committee to coerce a messy text blob into strict JSON matching expected_schema.

    This helper is intended for planner-style function calling where we require a JSON object
    with a known shape (e.g., steps[], action/rationale/patch_plan). It routes the raw text
    through a dedicated JSON-fixer prompt and then parses the result with JSONParser.
    """
    schema_desc = json.dumps(expected_schema, default=str, ensure_ascii=False)
    sys_msg = (
        "You are JSONFixer. You receive messy AI output and MUST respond ONLY with strict JSON "
        "that matches this schema:\n"
        f"{schema_desc}\n"
        "Do not add any explanations, comments, or extra keys. If fields are missing, fill them "
        "with sensible defaults of the correct type. Your entire reply must be a single JSON object."
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": raw_text or ""},
    ]
    env = await committee_ai_text(
        messages,
        trace_id=f"{trace_id or 'committee'}.jsonify",
        rounds=rounds,
        temperature=temperature,
    )
    result_block = env.get("result") if isinstance(env, dict) else {}

    # Collect up to four candidate JSON texts: the merged text plus per-backend responses.
    candidates: List[str] = []
    if isinstance(result_block, dict):
        txt_main = result_block.get("text")
        if isinstance(txt_main, str) and txt_main.strip():
            candidates.append(txt_main)
        for key in ("synth", "qwen", "glm", "deepseek"):
            inner = result_block.get(key)
            if isinstance(inner, dict):
                resp = inner.get("response")
                if isinstance(resp, str) and resp.strip():
                    candidates.append(resp)
    # Always include the original raw_text as a last-resort candidate.
    if isinstance(raw_text, str) and raw_text.strip():
        candidates.append(raw_text)

    parser = JSONParser()
    parsed_candidates: List[Dict[str, Any]] = []
    for txt in candidates:
        obj = parser.parse(txt or "{}", expected_schema)
        if isinstance(obj, dict):
            parsed_candidates.append(obj)

    # Merge candidates field-wise, preferring the first non-empty value across candidates.
    def _default_for(expected: Any) -> Any:
        if expected is int:
            return 0
        if expected is float:
            return 0.0
        if expected is list:
            return []
        if expected is dict:
            return {}
        return ""

    merged: Dict[str, Any] = {}
    if isinstance(expected_schema, dict):
        for key, expected_type in expected_schema.items():
            value_set = False
            for cand in parsed_candidates:
                v = cand.get(key)
                if isinstance(expected_type, list):
                    if isinstance(v, list) and v:
                        merged[key] = v
                        value_set = True
                        break
                elif isinstance(expected_type, dict):
                    if isinstance(v, dict) and v:
                        merged[key] = v
                        value_set = True
                        break
                else:
                    if isinstance(v, expected_type):
                        merged[key] = v
                        value_set = True
                        break
            if not value_set:
                merged[key] = _default_for(expected_type)
    else:
        # Non-dict schema: just return the first successful parse or a default.
        if parsed_candidates:
            merged = parsed_candidates[0]
        else:
            merged = parser.parse("{}", expected_schema) if isinstance(expected_schema, dict) else {}

    emit_trace(
        STATE_DIR,
        str(trace_id or "committee"),
        "committee.jsonify.merge",
        {
            "trace_id": str(trace_id or "committee"),
            "candidates": len(candidates),
            "parsed_candidates": len(parsed_candidates),
        },
    )
    return merged


class CommitteeClient:
    """
    Thin, async-only façade that exposes the committee path as a method.
    """

    async def run(
        self,
        messages: List[Dict[str, Any]],
        trace_id: str,
        rounds: int | None = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        return await committee_ai_text(
            messages,
            trace_id=trace_id,
            rounds=rounds,
            temperature=temperature,
        )
