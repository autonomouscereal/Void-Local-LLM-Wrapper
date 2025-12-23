'''
committee_client.py: Committee client for text generation.
'''

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


def _env_int(name: str, default: int, *, min_val: int | None = None, max_val: int | None = None) -> int:
    raw = os.getenv(name, None)
    val = int(str(raw).strip()) if raw is not None else int(default)
    if min_val is not None and val < min_val:
        val = min_val
    if max_val is not None and val > max_val:
        val = max_val
    return val


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


def _is_empty_song_candidate(obj: Any) -> bool:
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
    if (not isinstance(bpm, (int, float)) or float(bpm) == 0.0) and not sections and not voices and not instruments and not motifs and not lyrics_sections:
        return True
    return False


QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11435")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen3:30b-a3b-instruct-2507-q4_K_M")
GLM_OLLAMA_BASE_URL = os.getenv("GLM_OLLAMA_BASE_URL", "http://localhost:11433")
GLM_MODEL_ID = os.getenv("GLM_MODEL_ID", "glm4:9b")
DEEPSEEK_CODER_OLLAMA_BASE_URL = os.getenv("DEEPSEEK_CODER_OLLAMA_BASE_URL", "http://localhost:11436")
DEEPSEEK_CODER_MODEL_ID = os.getenv("DEEPSEEK_CODER_MODEL_ID", "deepseek-coder-v2:lite")

DEFAULT_NUM_CTX = _env_int("DEFAULT_NUM_CTX", 8192, min_val=1024, max_val=262144)
COMMITTEE_MODEL_ID = os.getenv("COMMITTEE_MODEL_ID") or f"committee:{QWEN_MODEL_ID}+{GLM_MODEL_ID}+{DEEPSEEK_CODER_MODEL_ID}"
DEFAULT_COMMITTEE_ROUNDS = _env_int("COMMITTEE_ROUNDS", 1, min_val=1, max_val=10)

PARTICIPANT_MODELS: Dict[str, Dict[str, str]] = {
    "qwen": {"base": QWEN_BASE_URL, "model": QWEN_MODEL_ID},
    "glm": {"base": GLM_OLLAMA_BASE_URL, "model": GLM_MODEL_ID},
    "deepseek": {"base": DEEPSEEK_CODER_OLLAMA_BASE_URL, "model": DEEPSEEK_CODER_MODEL_ID},
}
COMMITTEE_PARTICIPANTS = [
    {"id": name, "base": cfg["base"], "model": cfg["model"]}
    for name, cfg in PARTICIPANT_MODELS.items()
]

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
STATE_DIR = os.path.join(UPLOAD_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)


async def call_ollama(base_url: str, payload: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    t_all = time.monotonic()
    model = payload.get("model")
    prompt = payload.get("prompt")
    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    log.info(
        f"[committee] ollama.call.start trace_id={trace_id} base={base_url} model={model} "
        f"stream={bool(payload.get('stream', False))} options={options} prompt_chars={len(str(prompt or ''))}"
    )

    emit_trace(
        STATE_DIR,
        trace_id,
        "committee.ollama.request",
        {
            "trace_id": trace_id,
            "base_url": base_url,
            "model": model,
            "payload": payload,
        },
    )

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        log.info(f"[committee] ollama.request base={base_url} model={model} trace_id={trace_id} prompt={prompt}")

        t_http = time.monotonic()
        resp = await client.post(f"{base_url.rstrip('/')}/api/generate", json=dict(payload))
        raw_text = resp.text or ""
        status_code = int(resp.status_code)

        log.info(
            f"[committee] ollama.http.response trace_id={trace_id} base={base_url} model={model} status={status_code} "
            f"http_dur_ms={int((time.monotonic() - t_http) * 1000.0)} content_type={resp.headers.get('content-type')} raw={raw_text}"
        )

        parser = JSONParser()
        t_parse = time.monotonic()
        parsed_obj = parser.parse(raw_text or "{}", {})
        parsed = parsed_obj if isinstance(parsed_obj, dict) else {}

        log.info(
            f"[committee] ollama.parsed trace_id={trace_id} base={base_url} model={model} status={status_code} "
            f"parse_dur_ms={int((time.monotonic() - t_parse) * 1000.0)} parsed={json.dumps(parsed, ensure_ascii=False, default=str)}"
        )

        if parser.errors:
            log.warning(
                f"[committee] ollama.parser_errors trace_id={trace_id} base={base_url} model={model} status={status_code} "
                f"last_error={parser.last_error} errors={json.dumps(list(parser.errors), ensure_ascii=False, default=str)}"
            )

        if not (200 <= status_code < 300):
            err_text = parsed.get("error") if isinstance(parsed.get("error"), str) else raw_text
            emit_trace(
                STATE_DIR,
                trace_id,
                "committee.ollama.http_error",
                {"trace_id": trace_id, "base_url": base_url, "status_code": status_code, "error": err_text, "raw": raw_text, "parsed": parsed},
            )
            log.error(
                f"[committee] ollama.http_error trace_id={trace_id} base={base_url} model={model} status={status_code} "
                f"dur_ms={int((time.monotonic() - t_all) * 1000.0)} error={err_text}"
            )
            return {
                "ok": False,
                "error": {
                    "code": "ollama_http_error",
                    "message": f"ollama returned HTTP {status_code}",
                    "status": status_code,
                    "base_url": base_url,
                    "details": {"error": err_text},
                },
            }

        response_str = parsed.get("response") if isinstance(parsed.get("response"), str) else ""
        response_str = _sanitize_mojibake_text(response_str or "")

        prompt_eval_val = parsed.get("prompt_eval_count")
        eval_count_val = parsed.get("eval_count")
        prompt_eval = int(prompt_eval_val) if isinstance(prompt_eval_val, (int, float)) else 0
        eval_count = int(eval_count_val) if isinstance(eval_count_val, (int, float)) else 0

        data: Dict[str, Any] = {"ok": True, "response": response_str, "prompt_eval_count": prompt_eval, "eval_count": eval_count}

        usage: Dict[str, int] | None = None
        if prompt_eval or eval_count:
            usage = {"prompt_tokens": int(prompt_eval), "completion_tokens": int(eval_count)}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            data["_usage"] = usage

        log.info(
            f"[committee] ollama.response base={base_url} model={model} trace_id={trace_id} status={status_code} response={response_str} usage={usage}"
        )
        log.info(
            f"[committee] ollama.call.finish trace_id={trace_id} ok=true dur_ms={int((time.monotonic() - t_all) * 1000.0)} "
            f"status={status_code} usage={usage} response={response_str}"
        )

        emit_trace(
            STATE_DIR,
            trace_id,
            "committee.ollama.response",
            {"trace_id": trace_id, "status_code": status_code, "usage": usage or {}, "payload": payload, "raw": raw_text, "parsed": parsed, "response_text": response_str, "response": data},
        )

        return data


def build_ollama_payload(messages: List[Dict[str, Any]], model: str, num_ctx: int, temperature: float) -> Dict[str, Any]:
    t0 = time.monotonic()
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

    max_chars = DEFAULT_NUM_CTX * 4
    if len(prompt) > max_chars:
        trimmed = prompt[-max_chars:]
        log.info(
            f"[committee] prompt.truncated model={model} num_ctx={int(DEFAULT_NUM_CTX)} orig_chars={len(prompt)} kept_chars={len(trimmed)}"
        )
        prompt = trimmed

    log.info(
        f"[committee] ollama.payload.rendered model={str(model)} num_ctx={int(num_ctx)} temperature={float(temperature)} "
        f"messages={len(messages or [])} prompt_chars={len(prompt)} dur_ms={int((time.monotonic() - t0) * 1000.0)}"
    )

    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "24h",
        "options": {
            "num_ctx": int(num_ctx),
            "temperature": float(temperature),
        },
    }


def _participant_for(member_id: str) -> Dict[str, str] | None:
    cfg = PARTICIPANT_MODELS.get(str(member_id or "").strip())
    return cfg if isinstance(cfg, dict) else None


async def committee_member_text(
    member_id: str,
    messages: List[Dict[str, Any]],
    *,
    trace_id: str,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    cfg = _participant_for(member_id=member_id)
    if not cfg:
        log.error(
            f"[committee] member.finish trace_id={trace_id} member={str(member_id or '')} ok=false dur_ms={int((time.monotonic() - t0) * 1000.0)} error=unknown_member"
        )
        return {"ok": False, "error": {"code": "unknown_member", "message": f"unknown committee member: {member_id}"}}

    base = (cfg.get("base") or "").rstrip("/") or QWEN_BASE_URL
    model = cfg.get("model") or QWEN_MODEL_ID

    log.info(
        f"[committee] member.start trace_id={trace_id} member={str(member_id or '')} base={str(base)} model={str(model)} "
        f"messages={len(messages or [])} temperature={float(temperature)}"
    )

    payload = build_ollama_payload(messages=messages or [], model=model, num_ctx=DEFAULT_NUM_CTX, temperature=temperature)
    res = await call_ollama(base_url=base, payload=payload, trace_id=trace_id)

    res["_base_url"] = base
    res["_model"] = model
    res["_member"] = str(member_id or "")

    log.info(
        f"[committee] member.finish trace_id={trace_id} member={str(member_id or '')} ok={bool(res.get('ok', True))} "
        f"dur_ms={int((time.monotonic() - t0) * 1000.0)} response_chars={len(str(res.get('response') or ''))} has_error={bool(res.get('error'))}"
    )
    return res


async def committee_synth_text(
    messages: List[Dict[str, Any]],
    *,
    trace_id: str,
    temperature: float = 0.0,
    synth_member: str = "qwen",
) -> Dict[str, Any]:
    log.info(
        f"[committee] synth.start trace_id={trace_id} synth_member={str(synth_member or 'qwen')} messages={len(messages or [])} temperature={float(temperature)}"
    )
    return await committee_member_text(
        member_id=synth_member,
        messages=messages,
        trace_id=trace_id,
        temperature=temperature,
    )


async def committee_ai_text(
    messages: List[Dict[str, Any]],
    *,
    trace_id: str,
    rounds: int | None = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    effective_rounds = int(rounds if rounds is not None else DEFAULT_COMMITTEE_ROUNDS)
    effective_rounds = max(1, int(effective_rounds))

    _temp_run = float(temperature)
    if _temp_run < 0.0:
        _temp_run = 0.0
    if _temp_run > 2.0:
        _temp_run = 2.0

    member_ids = [p.get("id") or "member" for p in COMMITTEE_PARTICIPANTS]

    answers: Dict[str, str] = {}
    critiques: Dict[str, str] = {}
    member_results: Dict[str, Dict[str, Any]] = {}
    critique_results: Dict[str, Dict[str, Any]] = {}

    t0 = time.monotonic()
    log.info(
        f"[committee] run.start trace_id={trace_id} rounds={int(effective_rounds)} temperature={float(_temp_run)} participants={member_ids} messages={len(messages or [])}"
    )

    emit_trace(
        STATE_DIR,
        trace_id,
        "committee.start",
        {
            "trace_id": trace_id,
            "rounds": int(effective_rounds),
            "temperature": float(_temp_run),
            "participants": member_ids,
            "messages": messages or [],
        },
    )

    for r in range(effective_rounds):
        log.info(f"[committee] round.start trace_id={trace_id} round={int(r + 1)}")

        for mid in member_ids:
            log.info(f"[committee] member_draft.start trace_id={trace_id} round={int(r + 1)} member={mid}")
            draft_lines = [
                f"You are committee member {mid}. Provide your best answer to the user.",
                "All content must be written in English only. Do NOT respond in any other language.",
            ]
            member_msgs = list(messages or []) + [{"role": "system", "content": "\n\n".join(draft_lines)}]
            emit_trace(
                STATE_DIR,
                trace_id,
                "committee.member_draft",
                {"trace_id": trace_id, "round": int(r + 1), "member": mid},
            )
            res = await committee_member_text(
                member_id=mid,
                messages=member_msgs,
                trace_id=trace_id,
                temperature=_temp_run,
            )
            member_results[mid] = res if isinstance(res, dict) else {}
            txt = ""
            if isinstance(res, dict) and res.get("ok") is not False:
                txt = str(res.get("response") or "").strip()
            answers[mid] = txt
            log.info(
                f"[committee] member_draft.finish trace_id={trace_id} round={int(r + 1)} member={mid} "
                f"ok={bool(isinstance(res, dict) and res.get('ok', True))} answer_chars={len(txt)} has_error={bool(isinstance(res, dict) and res.get('error'))}"
            )

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

            critique_lines = [
                f"You are committee member {mid}. This is cross-critique for debate round {r + 1}.",
                "Critique the other answers. Identify concrete mistakes, missing considerations, and specific improvements.",
                "Return a concise bullet list.",
            ]
            if other_blocks:
                critique_lines.append("\n\n".join(other_blocks))
            critique_lines.append("All content must be written in English only. Do NOT respond in any other language.")
            member_msgs = list(messages or []) + [{"role": "system", "content": "\n\n".join(critique_lines)}]
            emit_trace(
                STATE_DIR,
                trace_id,
                "committee.member_critique",
                {"trace_id": trace_id, "round": int(r + 1), "member": mid},
            )
            res = await committee_member_text(
                member_id=mid,
                messages=member_msgs,
                trace_id=trace_id,
                temperature=_temp_run,
            )
            critique_results[mid] = res if isinstance(res, dict) else {}
            crit_txt = ""
            if isinstance(res, dict) and res.get("ok") is not False:
                crit_txt = str(res.get("response") or "").strip()
            critiques[mid] = crit_txt
            others_count = sum(1 for oid in member_ids if oid != mid and (answers.get(oid) or "").strip())
            log.info(
                f"[committee] member_critique.finish trace_id={trace_id} round={int(r + 1)} member={mid} "
                f"ok={bool(isinstance(res, dict) and res.get('ok', True))} critique_chars={len(crit_txt)} "
                f"has_error={bool(isinstance(res, dict) and res.get('error'))} others_count={int(others_count)}"
            )

        critiques_chars = {mid: len(critiques.get(mid) or "") for mid in member_ids}
        log.info(f"[committee] round.revision.start trace_id={trace_id} round={int(r + 1)} critiques_chars={critiques_chars}")

        for mid in member_ids:
            log.info(f"[committee] member_revision.start trace_id={trace_id} round={int(r + 1)} member={mid}")
            ctx_lines: List[str] = [
                f"You are committee member {mid}. This is revision for debate round {r + 1}.",
                "Revise your answer using the critiques. Output ONLY your full final answer.",
            ]
            for oid in member_ids:
                if oid == mid:
                    continue
                otext = answers.get(oid) or ""
                if isinstance(otext, str) and otext.strip():
                    ctx_lines.append(f"Answer from {oid}:\n{otext.strip()}")

            crit_blocks: List[str] = []
            for cid in member_ids:
                ctext = critiques.get(cid) or ""
                if isinstance(ctext, str) and ctext.strip():
                    crit_blocks.append(f"Critique from {cid}:\n{ctext.strip()}")
            if crit_blocks:
                ctx_lines.append("\n\n".join(crit_blocks))

            prior = answers.get(mid) or ""
            if isinstance(prior, str) and prior.strip():
                ctx_lines.append(f"Your current answer is:\n{prior.strip()}")

            ctx_lines.append("All content must be written in English only. Do NOT respond in any other language.")
            member_msgs = list(messages or []) + [{"role": "system", "content": "\n\n".join(ctx_lines)}]
            emit_trace(
                STATE_DIR,
                trace_id,
                "committee.member_revision",
                {"trace_id": trace_id, "round": int(r + 1), "member": mid},
            )
            res = await committee_member_text(
                member_id=mid,
                messages=member_msgs,
                trace_id=trace_id,
                temperature=_temp_run,
            )
            member_results[mid] = res if isinstance(res, dict) else {}
            txt = ""
            if isinstance(res, dict) and res.get("ok") is not False:
                txt = str(res.get("response") or "").strip()
            answers[mid] = txt
            log.info(
                f"[committee] member_revision.finish trace_id={trace_id} round={int(r + 1)} member={mid} "
                f"ok={bool(isinstance(res, dict) and res.get('ok', True))} answer_chars={len(txt)} has_error={bool(isinstance(res, dict) and res.get('error'))}"
            )

    member_summaries: List[Dict[str, Any]] = [{"member": mid, "answer": (answers.get(mid) or "")} for mid in member_ids]
    critique_summaries: List[Dict[str, Any]] = [{"member": mid, "critique": (critiques.get(mid) or "")} for mid in member_ids]

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

    emit_trace(
        STATE_DIR,
        trace_id,
        "committee.synth.request",
        {"trace_id": trace_id, "synth_member": "qwen"},
    )

    t_synth = time.monotonic()
    synth_env = await committee_synth_text(
        messages=synth_messages,
        trace_id=trace_id,
        temperature=0.0,
        synth_member="qwen",
    )
    synth_text = ""
    if isinstance(synth_env, dict) and synth_env.get("ok") is not False:
        synth_text = str(synth_env.get("response") or "").strip()

    log.info(
        f"[committee] synth.finish trace_id={trace_id} ok={bool(isinstance(synth_env, dict) and synth_env.get('ok', True))} "
        f"dur_ms={int((time.monotonic() - t_synth) * 1000.0)} synth_chars={len(synth_text)} has_error={bool(isinstance(synth_env, dict) and synth_env.get('error'))}"
    )

    final_text = synth_text
    if not final_text:
        best_mid = None
        best_len = 0
        for mid in member_ids:
            ans = answers.get(mid) or ""
            if not isinstance(ans, str):
                continue
            l = len(ans.strip())
            if l > best_len:
                best_len = l
                best_mid = mid
        if best_mid:
            final_text = (answers.get(best_mid) or "").strip()
            log.warning(
                f"[committee] synth.fallback trace_id={trace_id} chosen_member={str(best_mid)} chosen_chars={len(final_text)}"
            )

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

    emit_trace(
        STATE_DIR,
        trace_id,
        "committee.finish",
        {"trace_id": trace_id, "ok": ok, "final_text": final_text, "members": member_summaries, "critiques": critique_summaries},
    )

    log.info(
        f"[committee] run.finish trace_id={trace_id} ok={ok} dur_ms={int((time.monotonic() - t0) * 1000.0)} "
        f"final_chars={len(final_text or '')} backend_errors={backend_errors}"
    )

    if not ok:
        members_status = {
            mid: {
                "ok": bool(isinstance(mres, dict) and mres.get("ok", True)),
                "has_error": bool(isinstance(mres, dict) and mres.get("error")),
                "error": (mres.get("error") if isinstance(mres, dict) else None),
            }
            for mid, mres in member_results.items()
        }
        log.error(f"[committee] no_answer trace_id={trace_id} backend_errors={backend_errors} members={members_status}")

    result_payload: Dict[str, Any] = {
        "text": final_text,
        "members": member_summaries,
        "critiques": critique_summaries,
        "backend_errors": backend_errors,
        "qwen": member_results.get("qwen") or {},
        "glm": member_results.get("glm") or {},
        "deepseek": member_results.get("deepseek") or {},
        "synth": (dict(synth_env) if isinstance(synth_env, dict) else {"response": final_text}),
        "critique_envelopes": critique_results,
    }

    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "ok": ok,
        "result": result_payload if ok else None,
        "error": None
        if ok
        else {
            "code": "committee_no_answer",
            "message": "Committee did not produce a non-empty answer",
            "backend_errors": backend_errors,
        },
    }


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


def _is_default_scalar(val: Any, typ: Any) -> bool:
    if typ is int:
        return not isinstance(val, int) or int(val) == 0
    if typ is float:
        return not isinstance(val, (int, float)) or float(val) == 0.0
    if typ is str:
        return not isinstance(val, str) or not val.strip()
    return val is None


async def committee_jsonify(
    raw_text: str,
    expected_schema: Any,
    *,
    trace_id: str,
    rounds: int | None = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    schema_template = _schema_to_template(expected=expected_schema)
    schema_desc = json.dumps(schema_template, ensure_ascii=False)

    emit_trace(
        STATE_DIR,
        trace_id,
        "committee.jsonify.start",
        {
            "trace_id": trace_id,
            "schema_preview": schema_desc[:600],
            "schema_len": len(schema_desc),
            "raw_preview": (raw_text or "")[:600],
            "raw_len": len(raw_text or ""),
        },
    )

    sys_msg = (
        "You are JSONFixer. You receive messy AI output and MUST respond with exactly ONE JSON object.\n"
        "Respond ONLY in English. NO other languages are allowed.\n"
        "NO markdown, NO code fences, NO prose, NO comments.\n\n"
        "The JSON MUST match this exact structure (keys and nesting):\n\n"
        f"{schema_desc}\n\n"
        "Rules:\n"
        "- You MUST preserve all keys and their nesting exactly as shown.\n"
        "- You MUST NOT add any extra keys at any level.\n"
        "- You MUST NOT remove or rename any keys.\n"
        "- Values MUST be of the correct JSON type implied by the structure (number, string, object, array, null).\n"
        "- The output MUST be valid JSON:\n"
        "  - All keys and string values in double quotes.\n"
        "  - No trailing commas.\n"
        "  - No unescaped quotes inside strings.\n"
        "- You MUST NOT output any text before or after the JSON object.\n"
        "- Ignore any apology, refusal, or capability disclaimer in the input; your ONLY job is to produce valid JSON.\n"
        "- If the input is pure prose or contains multiple candidates, extract or reconstruct the best JSON candidate that fits the schema.\n\n"
        "Fill in this JSON object according to the provided text. Respond ONLY with the JSON object."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": raw_text or ""},
    ]

    env = await committee_ai_text(
        messages=messages,
        trace_id=trace_id,
        rounds=rounds,
        temperature=temperature,
    )
    result_block = env.get("result") if isinstance(env, dict) else {}

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
    if isinstance(raw_text, str) and raw_text.strip():
        candidates.append(raw_text)

    emit_trace(
        STATE_DIR,
        trace_id,
        "committee.jsonify.candidates",
        {
            "trace_id": trace_id,
            "count": len(candidates),
            "candidates": candidates,
        },
    )

    parsed_candidates: List[Dict[str, Any]] = []
    song_schema: Dict[str, Any] | None = None
    if isinstance(expected_schema, dict):
        maybe_song = expected_schema.get("song")
        if isinstance(maybe_song, dict) and "global" in maybe_song and "sections" in maybe_song:
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
            if not isinstance(song_obj, dict) or _is_empty_song_candidate(song_obj):
                continue
            sections = song_obj.get("sections") if isinstance(song_obj.get("sections"), list) else []
            voices = song_obj.get("voices") if isinstance(song_obj.get("voices"), list) else []
            instruments = song_obj.get("instruments") if isinstance(song_obj.get("instruments"), list) else []
            motifs = song_obj.get("motifs") if isinstance(song_obj.get("motifs"), list) else []
            num_sections = len(sections)
            richness = len(voices) + len(instruments) + len(motifs)
            rich.append((idx, num_sections, richness))
        if rich:
            rich_sorted = sorted(rich, key=lambda t: (-t[1], -t[2], t[0]))
            best_idx = rich_sorted[0][0]
            reordered: List[Dict[str, Any]] = []
            reordered.append(parsed_candidates[best_idx])
            for i, cand in enumerate(parsed_candidates):
                if i == best_idx:
                    continue
                song_obj = cand.get("song")
                if isinstance(song_obj, dict) and not _is_empty_song_candidate(song_obj):
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
                        if isinstance(v, dict) and not _is_empty_song_candidate(v):
                            merged[key] = v
                            value_set = True
                            break
                    else:
                        if isinstance(v, dict) and v:
                            merged[key] = v
                            value_set = True
                            break
                else:
                    if isinstance(v, expected_type) and not _is_default_scalar(v, expected_type):
                        merged[key] = v
                        value_set = True
                        break
            if not value_set:
                merged[key] = _default_for(expected_type)
    else:
        if parsed_candidates:
            merged = parsed_candidates[0]
        else:
            if isinstance(expected_schema, list):
                merged = []
            else:
                merged = _default_for(expected_schema)

    emit_trace(
        STATE_DIR,
        trace_id,
        "committee.jsonify.merge",
        {
            "trace_id": trace_id,
            "candidates": len(candidates),
            "parsed_candidates": len(parsed_candidates),
            "merged_keys": sorted(list(merged.keys())) if isinstance(merged, dict) else [],
        },
    )

    log.info(
        f"[committee.jsonify] finish trace_id={trace_id} dur_ms={int((time.monotonic() - t0) * 1000.0)} "
        f"candidates={int(len(candidates))} parsed_candidates={int(len(parsed_candidates))} merged_is_dict={bool(isinstance(merged, dict))} "
        f"merged_keys={(sorted(list(merged.keys())) if isinstance(merged, dict) else [])}"
    )

    return merged
    