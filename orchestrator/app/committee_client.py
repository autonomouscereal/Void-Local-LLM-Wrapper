"""
committee_client.py: Committee client for text generation (CHAT /api/chat).
- Uses Ollama /api/chat (messages, not prompt)
- Keeps full committee flow (draft -> critique -> revision -> synth)
- Keeps logging + emit_trace
- NO try/except added (no new error handling)
- temperature optional (omit from options when None)
- rounds optional
- All internal calls use keyword args (no positional)
- Function defs are single-line
"""

from __future__ import annotations

from typing import Any, Dict, List
import logging
import os
import json
import time

import httpx  # type: ignore

from .json_parser import JSONParser
from .trace_utils import emit_trace

log = logging.getLogger(__name__)

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11435")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen3:30b-a3b-instruct-2507-q4_K_M")
GLM_OLLAMA_BASE_URL = os.getenv("GLM_OLLAMA_BASE_URL", "http://localhost:11433")
GLM_MODEL_ID = os.getenv("GLM_MODEL_ID", "glm4:9b")
DEEPSEEK_CODER_OLLAMA_BASE_URL = os.getenv("DEEPSEEK_CODER_OLLAMA_BASE_URL", "http://localhost:11436")
DEEPSEEK_CODER_MODEL_ID = os.getenv("DEEPSEEK_CODER_OLLAMA_BASE_URL", None) and os.getenv("DEEPSEEK_CODER_MODEL_ID", "deepseek-coder-v2:lite") or os.getenv("DEEPSEEK_CODER_MODEL_ID", "deepseek-coder-v2:lite")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
STATE_DIR = os.path.join(UPLOAD_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)


def _env_int(env_var_name: str, default: int, *, min_val: int | None = None, max_val: int | None = None) -> int:
    raw = os.getenv(env_var_name, None)
    val = int(str(raw).strip()) if raw is not None else int(default)
    if min_val is not None and val < min_val:
        val = min_val
    if max_val is not None and val > max_val:
        val = max_val
    return val


DEFAULT_NUM_CTX = _env_int("DEFAULT_NUM_CTX", 8192, min_val=1024, max_val=262144)
DEFAULT_COMMITTEE_ROUNDS = _env_int("COMMITTEE_ROUNDS", 1, min_val=1, max_val=10)
COMMITTEE_MODEL_ID = os.getenv("COMMITTEE_MODEL_ID") or f"committee:{QWEN_MODEL_ID}+{GLM_MODEL_ID}+{DEEPSEEK_CODER_MODEL_ID}"

PARTICIPANT_MODELS = {"qwen": {"base": QWEN_BASE_URL, "model": QWEN_MODEL_ID}, "glm": {"base": GLM_OLLAMA_BASE_URL, "model": GLM_MODEL_ID}, "deepseek": {"base": DEEPSEEK_CODER_OLLAMA_BASE_URL, "model": DEEPSEEK_CODER_MODEL_ID}}
COMMITTEE_PARTICIPANTS = [{"id": member_id, "base": cfg["base"], "model": cfg["model"]} for member_id, cfg in PARTICIPANT_MODELS.items()]


def _sanitize_mojibake_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    replacements = (("Ã¢Â\x80Â\x99", "'"), ("Ã¢ÂÂ", "'"), ("â\x80\x99", "'"), ("â€™", "'"), ("â", "'"))
    out = text
    for bad, good in replacements:
        if bad in out:
            out = out.replace(bad, good)
    return out


def _participant_for(member_id: str):
    cfg = PARTICIPANT_MODELS.get(str(member_id or "").strip())
    return cfg if isinstance(cfg, dict) else None


def _normalize_messages(messages: List[Dict[str, Any]]):
    out: List[Dict[str, Any]] = []
    for message_entry in (messages or []):
        if not isinstance(message_entry, dict):
            continue
        role = str(message_entry.get("role") or "user")
        tool_call_name = message_entry.get("tool_call_name") if isinstance(message_entry.get("tool_call_name"), str) else None
        content_val = message_entry.get("content")
        if isinstance(content_val, list):
            text_parts: List[str] = []
            for part in content_val:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(str(part.get("text") or ""))
            content = "\n".join(text_parts)
        else:
            content = str(content_val or "")
        msg: Dict[str, Any] = {"role": role, "content": content}
        if role == "tool" and isinstance(tool_call_name, str) and tool_call_name.strip():
            msg["name"] = tool_call_name.strip()
        out.append(msg)
    return out


def build_ollama_payload(messages: List[Dict[str, Any]], model: str, num_ctx: int, temperature: float | None = None):
    t0 = time.monotonic()
    norm_messages = _normalize_messages(messages=messages or [])
    options: Dict[str, Any] = {"num_ctx": int(num_ctx)}
    if temperature is not None:
        options["temperature"] = float(temperature)
    payload: Dict[str, Any] = {"model": str(model), "messages": norm_messages, "stream": False, "keep_alive": "24h", "options": options}
    # Full-fidelity: log the entire payload (messages + options) for debugging and distillation.
    log.info(
        "[committee] ollama.payload.rendered model=%s num_ctx=%d temperature=%r messages=%d dur_ms=%d payload=%s",
        str(model),
        int(num_ctx),
        (None if temperature is None else float(temperature)),
        len(norm_messages),
        int((time.monotonic() - t0) * 1000.0),
        json.dumps(payload, ensure_ascii=False, default=str),
    )
    return payload


async def call_ollama(base_url: str, payload: Dict[str, Any], trace_id: str):
    t_all = time.monotonic()
    model = payload.get("model")
    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    log.info(
        "[committee] ollama.call.start trace_id=%s base=%s model=%r stream=%s options=%s messages=%d payload=%s",
        trace_id,
        base_url,
        model,
        bool(payload.get("stream", False)),
        options,
        len(messages),
        json.dumps(payload, ensure_ascii=False, default=str),
    )
    emit_trace(
        state_dir=STATE_DIR,
        trace_id=trace_id,
        kind="committee.ollama.request",
        payload={"trace_id": trace_id, "base_url": base_url, "model": model, "payload": payload},
    )
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        t_http = time.monotonic()
        resp = await client.post(url=f"{base_url.rstrip('/')}/api/chat", json=dict(payload))
        raw_text = resp.text or ""
        status_code = int(resp.status_code)
        log.info(
            "[committee] ollama.http.response trace_id=%s base=%s model=%r status=%d http_dur_ms=%d content_type=%r raw=%s",
            trace_id,
            base_url,
            model,
            status_code,
            int((time.monotonic() - t_http) * 1000.0),
            resp.headers.get("content-type"),
            raw_text,
        )
        parser = JSONParser()
        t_parse = time.monotonic()
        try:
        parsed_obj = parser.parse(raw_text or "{}", {})
            if not isinstance(parsed_obj, dict):
                log.warning("[committee] ollama.parsed: JSONParser returned non-dict type=%s trace_id=%s base=%s model=%r", type(parsed_obj).__name__, trace_id, base_url, model)
                parsed_obj = {}
        except Exception as ex:
            log.warning("[committee] ollama.parse_exception trace_id=%s base=%s model=%r ex=%s raw_text_prefix=%s", trace_id, base_url, model, ex, (raw_text or "")[:200], exc_info=True)
            parsed_obj = {}
        parsed = parsed_obj if isinstance(parsed_obj, dict) else {}
        log.info(
            "[committee] ollama.parsed trace_id=%s base=%s model=%r status=%d parse_dur_ms=%d parsed=%s",
            trace_id,
            base_url,
            model,
            status_code,
            int((time.monotonic() - t_parse) * 1000.0),
            json.dumps(parsed, ensure_ascii=False, default=str),
        )
        if parser.errors:
            log.warning(
                "[committee] ollama.parser_errors trace_id=%s base=%s model=%r status=%d last_error=%r errors=%s",
                trace_id,
                base_url,
                model,
                status_code,
                parser.last_error,
                json.dumps(list(parser.errors), ensure_ascii=False, default=str),
            )
        if not (200 <= status_code < 300):
            err_text = parsed.get("error") if isinstance(parsed.get("error"), str) else raw_text
            emit_trace(
                state_dir=STATE_DIR,
                trace_id=trace_id,
                kind="committee.ollama.http_error",
                payload={"trace_id": trace_id, "base_url": base_url, "status_code": status_code, "error": err_text, "raw": raw_text, "parsed": parsed},
            )
            log.error(
                "[committee] ollama.http_error trace_id=%s base=%s model=%r status=%d dur_ms=%d error=%s raw=%s",
                trace_id,
                base_url,
                model,
                status_code,
                int((time.monotonic() - t_all) * 1000.0),
                err_text,
                raw_text,
            )
            return {"ok": False, "error": {"code": "ollama_http_error", "message": f"ollama returned HTTP {status_code}", "status": status_code, "base_url": base_url, "details": {"error": err_text}}}
        response_str = ""
        msg = parsed.get("message") if isinstance(parsed.get("message"), dict) else {}
        if isinstance(msg.get("content"), str):
            response_str = str(msg.get("content") or "")
        elif isinstance(parsed.get("response"), str):
            response_str = str(parsed.get("response") or "")
        response_str = _sanitize_mojibake_text(response_str or "")
        prompt_eval_val = parsed.get("prompt_eval_count")
        eval_count_val = parsed.get("eval_count")
        prompt_eval = int(prompt_eval_val) if isinstance(prompt_eval_val, (int, float)) else 0
        eval_count = int(eval_count_val) if isinstance(eval_count_val, (int, float)) else 0
        data: Dict[str, Any] = {"ok": True, "response": response_str, "prompt_eval_count": prompt_eval, "eval_count": eval_count}
        usage = None
        if prompt_eval or eval_count:
            usage = {"prompt_tokens": int(prompt_eval), "completion_tokens": int(eval_count)}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            data["_usage"] = usage
        log.info(
            "[committee] ollama.response base=%s model=%r trace_id=%s status=%d usage=%s response=%s",
            base_url,
            model,
            trace_id,
            status_code,
            usage,
            response_str,
        )
        log.info(
            "[committee] ollama.call.finish trace_id=%s ok=true dur_ms=%d status=%d usage=%s response=%s",
            trace_id,
            int((time.monotonic() - t_all) * 1000.0),
            status_code,
            usage,
            response_str,
        )
        emit_trace(
            state_dir=STATE_DIR,
            trace_id=trace_id,
            kind="committee.ollama.response",
            payload={
                "trace_id": trace_id,
                "status_code": status_code,
                "usage": (usage or {}),
                "payload": payload,
                "raw": raw_text,
                "parsed": parsed,
                "response_text": response_str,
                "response_obj": data,
            },
        )
        return data


async def committee_member_text(member_id: str, messages: List[Dict[str, Any]], *, trace_id: str, temperature: float | None = None):
    t0 = time.monotonic()
    cfg = _participant_for(member_id=member_id)
    if not cfg:
        log.error(f"[committee] member.finish trace_id={trace_id} member={str(member_id or '')} ok=false dur_ms={int((time.monotonic() - t0) * 1000.0)} error=unknown_member")
        return {"ok": False, "error": {"code": "unknown_member", "message": f"unknown committee member: {member_id}"}}
    base = (cfg.get("base") or "").rstrip("/") or QWEN_BASE_URL
    model = cfg.get("model") or QWEN_MODEL_ID
    log.info(
        "[committee] member.start trace_id=%s member=%s base=%s model=%s temperature=%r messages=%d messages_full=%s",
        trace_id,
        str(member_id or ""),
        str(base),
        str(model),
        (None if temperature is None else float(temperature)),
        len(messages or []),
        json.dumps(messages or [], ensure_ascii=False, default=str),
    )
    payload = build_ollama_payload(messages=messages or [], model=model, num_ctx=DEFAULT_NUM_CTX, temperature=temperature)
    res = await call_ollama(base_url=base, payload=payload, trace_id=trace_id)
    if isinstance(res, dict):
        res["_base_url"] = base
        res["_model"] = model
        res["_member"] = str(member_id or "")
    log.info(
        "[committee] member.finish trace_id=%s member=%s ok=%s dur_ms=%d result=%s",
        trace_id,
        str(member_id or ""),
        bool(isinstance(res, dict) and res.get("ok", True)),
        int((time.monotonic() - t0) * 1000.0),
        json.dumps(res if isinstance(res, dict) else {"_raw": res}, ensure_ascii=False, default=str),
    )
    return res


async def committee_synth_text(messages: List[Dict[str, Any]], *, trace_id: str, temperature: float | None = None, synth_member: str = "qwen"):
    log.info(f"[committee] synth.start trace_id={trace_id} synth_member={str(synth_member or 'qwen')} messages={len(messages or [])} temperature={(None if temperature is None else float(temperature))}")
    return await committee_member_text(member_id=synth_member, messages=messages, trace_id=trace_id, temperature=temperature)


async def committee_ai_text(messages: List[Dict[str, Any]], *, trace_id: str, rounds: int | None = None, temperature: float | None = None):
    effective_rounds = int(rounds if rounds is not None else DEFAULT_COMMITTEE_ROUNDS)
    effective_rounds = max(1, int(effective_rounds))
    temp_run = temperature
    if isinstance(temp_run, (int, float)):
        if float(temp_run) < 0.0:
            temp_run = 0.0
        if float(temp_run) > 2.0:
            temp_run = 2.0
        temp_run = float(temp_run)
    member_ids = [p.get("id") or "member" for p in COMMITTEE_PARTICIPANTS]
    answers: Dict[str, str] = {}
    critiques: Dict[str, str] = {}
    member_results: Dict[str, Dict[str, Any]] = {}
    critique_results: Dict[str, Dict[str, Any]] = {}
    t0 = time.monotonic()
    log.info(f"[committee] run.start trace_id={trace_id} rounds={int(effective_rounds)} temperature={(None if temp_run is None else float(temp_run))} participants={member_ids} messages={len(messages or [])}")
    emit_trace(state_dir=STATE_DIR, trace_id=trace_id, kind="committee.start", payload={"trace_id": trace_id, "rounds": int(effective_rounds), "temperature": (None if temp_run is None else float(temp_run)), "participants": member_ids, "messages": (messages or [])})
    for r in range(effective_rounds):
        log.info(f"[committee] round.start trace_id={trace_id} round={int(r + 1)}")
        for mid in member_ids:
            log.info(f"[committee] member_draft.start trace_id={trace_id} round={int(r + 1)} member={mid}")
            sys_txt = f"You are committee member {mid}. Provide your best answer to the user.\nAll content must be written in English only. Do NOT respond in any other language."
            member_msgs = list(messages or []) + [{"role": "system", "content": sys_txt}]
            emit_trace(state_dir=STATE_DIR, trace_id=trace_id, kind="committee.member_draft", payload={"trace_id": trace_id, "round": int(r + 1), "member": mid})
            res = await committee_member_text(member_id=mid, messages=member_msgs, trace_id=trace_id, temperature=temp_run)
            member_results[mid] = res if isinstance(res, dict) else {}
            txt = str(res.get("response") or "").strip() if isinstance(res, dict) and res.get("ok") is not False else ""
            answers[mid] = txt
            log.info(f"[committee] member_draft.finish trace_id={trace_id} round={int(r + 1)} member={mid} ok={bool(isinstance(res, dict) and res.get('ok', True))} answer_chars={len(txt)} has_error={bool(isinstance(res, dict) and res.get('error'))}")
        answers_chars = {mid: len(answers.get(mid) or "") for mid in member_ids}
        log.info(f"[committee] round.critique.start trace_id={trace_id} round={int(r + 1)} answers_chars={answers_chars}")
        critiques = {}
        for mid in member_ids:
            log.info(f"[committee] member_critique.start trace_id={trace_id} round={int(r + 1)} member={mid}")
            other_blocks: List[str] = []
            for oid in member_ids:
                if oid == mid:
                    continue
                otext = answers.get(oid) or ""
                if isinstance(otext, str) and otext.strip():
                    other_blocks.append(f"Answer from {oid}:\n{otext.strip()}")
            critique_txt = f"You are committee member {mid}. This is cross-critique for debate round {r + 1}.\nCritique the other answers. Identify concrete mistakes, missing considerations, and specific improvements.\nReturn a concise bullet list.\n" + ("\n\n".join(other_blocks) if other_blocks else "") + "\nAll content must be written in English only. Do NOT respond in any other language."
            member_msgs = list(messages or []) + [{"role": "system", "content": critique_txt}]
            emit_trace(state_dir=STATE_DIR, trace_id=trace_id, kind="committee.member_critique", payload={"trace_id": trace_id, "round": int(r + 1), "member": mid})
            res = await committee_member_text(member_id=mid, messages=member_msgs, trace_id=trace_id, temperature=temp_run)
            critique_results[mid] = res if isinstance(res, dict) else {}
            crit_txt = str(res.get("response") or "").strip() if isinstance(res, dict) and res.get("ok") is not False else ""
            critiques[mid] = crit_txt
            others_count = sum(1 for oid in member_ids if oid != mid and (answers.get(oid) or "").strip())
            log.info(f"[committee] member_critique.finish trace_id={trace_id} round={int(r + 1)} member={mid} ok={bool(isinstance(res, dict) and res.get('ok', True))} critique_chars={len(crit_txt)} has_error={bool(isinstance(res, dict) and res.get('error'))} others_count={int(others_count)}")
        critiques_chars = {mid: len(critiques.get(mid) or "") for mid in member_ids}
        log.info(f"[committee] round.revision.start trace_id={trace_id} round={int(r + 1)} critiques_chars={critiques_chars}")
        for mid in member_ids:
            log.info(f"[committee] member_revision.start trace_id={trace_id} round={int(r + 1)} member={mid}")
            ctx_lines: List[str] = []
            ctx_lines.append(f"You are committee member {mid}. This is revision for debate round {r + 1}.")
            ctx_lines.append("Revise your answer using the critiques. Output ONLY your full final answer.")
            for oid in member_ids:
                if oid == mid:
                    continue
                otext = answers.get(oid) or ""
                if isinstance(otext, str) and otext.strip():
                    ctx_lines.append(f"Answer from {oid}:\n{otext.strip()}")
            crit_blocks: List[str] = []
            for member_id in member_ids:
                ctext = critiques.get(member_id) or ""
                if isinstance(ctext, str) and ctext.strip():
                    crit_blocks.append(f"Critique from {member_id}:\n{ctext.strip()}")
            if crit_blocks:
                ctx_lines.append("\n\n".join(crit_blocks))
            prior = answers.get(mid) or ""
            if isinstance(prior, str) and prior.strip():
                ctx_lines.append(f"Your current answer is:\n{prior.strip()}")
            ctx_lines.append("All content must be written in English only. Do NOT respond in any other language.")
            member_msgs = list(messages or []) + [{"role": "system", "content": "\n\n".join(ctx_lines)}]
            emit_trace(state_dir=STATE_DIR, trace_id=trace_id, kind="committee.member_revision", payload={"trace_id": trace_id, "round": int(r + 1), "member": mid})
            res = await committee_member_text(member_id=mid, messages=member_msgs, trace_id=trace_id, temperature=temp_run)
            member_results[mid] = res if isinstance(res, dict) else {}
            txt = str(res.get("response") or "").strip() if isinstance(res, dict) and res.get("ok") is not False else ""
            answers[mid] = txt
            log.info(f"[committee] member_revision.finish trace_id={trace_id} round={int(r + 1)} member={mid} ok={bool(isinstance(res, dict) and res.get('ok', True))} answer_chars={len(txt)} has_error={bool(isinstance(res, dict) and res.get('error'))}")
    member_summaries = [{"member": mid, "answer": (answers.get(mid) or "")} for mid in member_ids]
    critique_summaries = [{"member": mid, "critique": (critiques.get(mid) or "")} for mid in member_ids]
    synth_parts: List[str] = []
    synth_parts.append("You are the committee synthesizer. Produce one final answer.")
    synth_parts.append("Rules: resolve conflicts by correctness; do not mention the committee or multiple models; output English only.")
    synth_parts.append("### Candidate Answers")
    for mid in member_ids:
        ans = answers.get(mid) or ""
        if isinstance(ans, str) and ans.strip():
            synth_parts.append(f"#### {mid}\n{ans.strip()}")
    if any(isinstance(critiques.get(mid), str) and str(critiques.get(mid)).strip() for mid in member_ids):
        synth_parts.append("### Critiques")
        for mid in member_ids:
            crit = critiques.get(mid) or ""
            if isinstance(crit, str) and crit.strip():
                synth_parts.append(f"#### {mid}\n{crit.strip()}")
    synth_parts.append("### Task\nSynthesize the best final answer. Output ONLY the final answer.")
    synth_messages = list(messages or []) + [{"role": "system", "content": "\n\n".join(synth_parts)}]
    emit_trace(state_dir=STATE_DIR, trace_id=trace_id, kind="committee.synth.request", payload={"trace_id": trace_id, "synth_member": "qwen"})
    t_synth = time.monotonic()
    synth_env = await committee_synth_text(messages=synth_messages, trace_id=trace_id, temperature=0.0, synth_member="qwen")
    synth_text = str(synth_env.get("response") or "").strip() if isinstance(synth_env, dict) and synth_env.get("ok") is not False else ""
    log.info(f"[committee] synth.finish trace_id={trace_id} ok={bool(isinstance(synth_env, dict) and synth_env.get('ok', True))} dur_ms={int((time.monotonic() - t_synth) * 1000.0)} synth_chars={len(synth_text)} has_error={bool(isinstance(synth_env, dict) and synth_env.get('error'))}")
    final_text = synth_text
    if not final_text:
        best_mid = None
        best_len = 0
        for mid in member_ids:
            ans = answers.get(mid) or ""
            if not isinstance(ans, str):
                log.debug("[committee] synth.fallback: skipping non-str answer member=%s type=%s trace_id=%s", mid, type(ans).__name__, trace_id)
                continue
            l = len(ans.strip())
            if l > best_len:
                best_len = l
                best_mid = mid
        if best_mid:
            final_text = (answers.get(best_mid) or "").strip()
            log.warning(f"[committee] synth.fallback trace_id={trace_id} chosen_member={str(best_mid)} chosen_chars={len(final_text)}")
    ok = bool(isinstance(final_text, str) and final_text.strip())
    backend_errors: List[Dict[str, Any]] = []
    for mid, mres in member_results.items():
        if isinstance(mres, dict) and mres.get("ok") is False and isinstance(mres.get("error"), dict):
            backend_errors.append({"member": mid, "phase": "revision", "error": mres.get("error")})
    for mid, cres in critique_results.items():
        if isinstance(cres, dict) and cres.get("ok") is False and isinstance(cres.get("error"), dict):
            backend_errors.append({"member": mid, "phase": "critique", "error": cres.get("error")})
    if isinstance(synth_env, dict) and synth_env.get("ok") is False and isinstance(synth_env.get("error"), dict):
        backend_errors.append({"member": "synth", "phase": "synth", "error": synth_env.get("error")})
    emit_trace(state_dir=STATE_DIR, trace_id=trace_id, kind="committee.finish", payload={"trace_id": trace_id, "ok": ok, "final_text": final_text, "members": member_summaries, "critiques": critique_summaries})
    log.info(f"[committee] run.finish trace_id={trace_id} ok={ok} dur_ms={int((time.monotonic() - t0) * 1000.0)} final_chars={len(final_text or '')} backend_errors={backend_errors}")
    if not ok:
        members_status = {mid: {"ok": bool(isinstance(mres, dict) and mres.get("ok", True)), "has_error": bool(isinstance(mres, dict) and mres.get("error")), "error": (mres.get("error") if isinstance(mres, dict) else None)} for mid, mres in member_results.items()}
        log.error(f"[committee] no_answer trace_id={trace_id} backend_errors={backend_errors} members={members_status}")
    result_payload: Dict[str, Any] = {"text": final_text, "members": member_summaries, "critiques": critique_summaries, "backend_errors": backend_errors, "qwen": (member_results.get("qwen") or {}), "glm": (member_results.get("glm") or {}), "deepseek": (member_results.get("deepseek") or {}), "synth": (dict(synth_env) if isinstance(synth_env, dict) else {"response": final_text}), "critique_envelopes": critique_results}
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "conversation_id": "",
        "ok": ok,
        "result": (result_payload if ok else None),
        "error": (None if ok else {"code": "committee_no_answer", "message": "Committee did not produce a non-empty answer", "backend_errors": backend_errors}),
    }


def _schema_to_template(expected: Any):
    if isinstance(expected, dict):
        return {k: _schema_to_template(expected=v) for k, v in expected.items()}
    if isinstance(expected, list):
        if not expected:
            return []
        return [_schema_to_template(expected=expected[0])]
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


def _default_for(expected: Any):
    if expected is int:
        return 0
    if expected is float:
        return 0.0
    if expected is list:
        return []
    if expected is dict:
        return {}
    return ""


def _is_default_scalar(val: Any, typ: Any):
    if typ is int:
        return (not isinstance(val, int)) or int(val) == 0
    if typ is float:
        return (not isinstance(val, (int, float))) or float(val) == 0.0
    if typ is str:
        return (not isinstance(val, str)) or (not val.strip())
    return val is None


def _is_empty_song_candidate(obj: Any):
    if not isinstance(obj, dict):
        return True
    global_block = obj.get("global") if isinstance(obj.get("global"), dict) else {}
    sections = obj.get("sections") if isinstance(obj.get("sections"), list) else []
    voices = obj.get("voices") if isinstance(obj.get("voices"), list) else []
    instruments = obj.get("instruments") if isinstance(obj.get("instruments"), list) else []
    motifs = obj.get("motifs") if isinstance(obj.get("motifs"), list) else []
    lyrics = obj.get("lyrics") if isinstance(obj.get("lyrics"), dict) else {}
    lyrics_sections = lyrics.get("sections") if isinstance(lyrics.get("sections"), list) else []
    bpm = global_block.get("bpm")
    if (not isinstance(bpm, (int, float)) or float(bpm) == 0.0) and (not sections) and (not voices) and (not instruments) and (not motifs) and (not lyrics_sections):
        return True
    return False


async def committee_jsonify(raw_text: str, expected_schema: Any, *, trace_id: str, rounds: int | None = None, temperature: float | None = None):
    t0 = time.monotonic()
    schema_template = _schema_to_template(expected=expected_schema)
    schema_desc = json.dumps(schema_template, ensure_ascii=False)
    emit_trace(state_dir=STATE_DIR, trace_id=trace_id, kind="committee.jsonify.start", payload={"trace_id": trace_id, "schema_preview": schema_desc[:600], "schema_len": len(schema_desc), "raw_preview": (raw_text or "")[:600], "raw_len": len(raw_text or "")})
    sys_msg = "You are JSONFixer. You receive messy AI output and MUST respond with exactly ONE JSON object.\nRespond ONLY in English. NO other languages are allowed.\nNO markdown, NO code fences, NO prose, NO comments.\n\nThe JSON MUST match this exact structure (keys and nesting):\n\n" + str(schema_desc) + "\n\nRules:\n- You MUST preserve all keys and their nesting exactly as shown.\n- You MUST NOT add any extra keys at any level.\n- You MUST NOT remove or rename any keys.\n- Values MUST be of the correct JSON type implied by the structure (number, string, object, array, null).\n- The output MUST be valid JSON:\n  - All keys and string values in double quotes.\n  - No trailing commas.\n  - No unescaped quotes inside strings.\n- You MUST NOT output any text before or after the JSON object.\n- Ignore any apology, refusal, or capability disclaimer in the input; your ONLY job is to produce valid JSON.\n- If the input is pure prose or contains multiple candidates, extract or reconstruct the best JSON candidate that fits the schema.\n\nFill in this JSON object according to the provided text. Respond ONLY with the JSON object."
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": (raw_text or "")}]
    env = await committee_ai_text(messages=messages, trace_id=trace_id, rounds=rounds, temperature=temperature)
    result_block = env.get("result") if isinstance(env, dict) else {}
    candidates: List[str] = []
    if isinstance(result_block, dict):
        txt_main = result_block.get("text")
        if isinstance(txt_main, str) and txt_main.strip():
            candidates.append(txt_main)
        for key in ("synth", "qwen", "glm", "deepseek"):
            inner = result_block.get(key)
            if isinstance(inner, dict) and isinstance(inner.get("response"), str) and str(inner.get("response") or "").strip():
                candidates.append(str(inner.get("response") or ""))
    if isinstance(raw_text, str) and raw_text.strip():
        candidates.append(raw_text)
    emit_trace(state_dir=STATE_DIR, trace_id=trace_id, kind="committee.jsonify.candidates", payload={"trace_id": trace_id, "count": len(candidates), "candidates": candidates})
    parsed_candidates: List[Dict[str, Any]] = []
    song_schema = None
    if isinstance(expected_schema, dict):
        maybe_song = expected_schema.get("song")
        if isinstance(maybe_song, dict) and ("global" in maybe_song) and ("sections" in maybe_song):
            song_schema = maybe_song
    for idx, txt in enumerate(candidates):
        clean_txt = _sanitize_mojibake_text(txt or "")
        parser = JSONParser()
        obj = parser.parse(clean_txt or "{}", expected_schema)
        if isinstance(obj, dict):
            parsed_candidates.append(obj)
    if song_schema is not None and parsed_candidates:
        rich: List[tuple[int, int, int]] = []
        for idx, cand in enumerate(parsed_candidates):
            song_obj = cand.get("song")
            if (not isinstance(song_obj, dict)) or _is_empty_song_candidate(obj=song_obj):
                continue
            sections = song_obj.get("sections") if isinstance(song_obj.get("sections"), list) else []
            voices = song_obj.get("voices") if isinstance(song_obj.get("voices"), list) else []
            instruments = song_obj.get("instruments") if isinstance(song_obj.get("instruments"), list) else []
            motifs = song_obj.get("motifs") if isinstance(song_obj.get("motifs"), list) else []
            num_sections = len(sections)
            richness = len(voices) + len(instruments) + len(motifs)
            rich.append((idx, num_sections, richness))
        if rich:
            ranked = [(-triple[1], -triple[2], triple[0]) for triple in rich]
            ranked.sort(reverse=True)
            best_idx = ranked[0][2]
            reordered: List[Dict[str, Any]] = []
            reordered.append(parsed_candidates[best_idx])
            for i, cand in enumerate(parsed_candidates):
                if i == best_idx:
                    continue
                song_obj = cand.get("song")
                if isinstance(song_obj, dict) and (not _is_empty_song_candidate(obj=song_obj)):
                    reordered.append(cand)
            parsed_candidates = reordered
    merged: Any = {}
    if isinstance(expected_schema, dict):
        schema_keys = set(expected_schema.keys())
        base = parsed_candidates[0] if parsed_candidates else {}
        merged = dict(base) if isinstance(base, dict) else {}
        for cand in parsed_candidates:
            if not isinstance(cand, dict):
                continue
            for k, v in cand.items():
                if k in schema_keys:
                    continue
                if k not in merged:
                    merged[k] = v
        for key, expected_type in expected_schema.items():
            value_set = False
            for cand in parsed_candidates:
                if not isinstance(cand, dict):
                    continue
                v = cand.get(key)
                if isinstance(expected_type, list):
                    if isinstance(v, list) and v:
                        merged[key] = v
                        value_set = True
                        break
                elif isinstance(expected_type, dict):
                    if key == "song" and song_schema is not None:
                        if isinstance(v, dict) and (not _is_empty_song_candidate(obj=v)):
                            merged[key] = v
                            value_set = True
                            break
                    else:
                        if isinstance(v, dict) and v:
                            merged[key] = v
                            value_set = True
                            break
                else:
                    if isinstance(v, expected_type) and (not _is_default_scalar(val=v, typ=expected_type)):
                        merged[key] = v
                        value_set = True
                        break
            if not value_set:
                merged[key] = _default_for(expected=expected_type)
    else:
        if parsed_candidates:
            merged = parsed_candidates[0]
        else:
            merged = [] if isinstance(expected_schema, list) else _default_for(expected=expected_schema)
    emit_trace(state_dir=STATE_DIR, trace_id=trace_id, kind="committee.jsonify.merge", payload={"trace_id": trace_id, "candidates": len(candidates), "parsed_candidates": len(parsed_candidates), "merged_keys": (sorted(list(merged.keys())) if isinstance(merged, dict) else [])})
    log.info(f"[committee.jsonify] finish trace_id={trace_id} dur_ms={int((time.monotonic() - t0) * 1000.0)} candidates={int(len(candidates))} parsed_candidates={int(len(parsed_candidates))} merged_is_dict={bool(isinstance(merged, dict))} merged_keys={(sorted(list(merged.keys())) if isinstance(merged, dict) else [])}")
    return merged
