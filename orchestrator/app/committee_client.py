from __future__ import annotations

from typing import Any, Dict, List
import logging

import os
import json
import time

import httpx  # type: ignore

from .json_parser import JSONParser
from .pipeline.compression_orchestrator import co_pack, frames_to_messages
from .trace_utils import emit_trace


def _env_int(name: str, default: int, *, min_val: int | None = None, max_val: int | None = None) -> int:
    """
    Parse an int env var defensively.

    IMPORTANT: This module is imported very early (and used by ICW/CO budgeting),
    so env parsing must never raise at import time.
    """
    raw = os.getenv(name, None)
    try:
        val = int(str(raw).strip()) if raw is not None else int(default)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.getLogger("orchestrator.committee").error("bad env %s=%r: %s", name, raw, exc, exc_info=True)
        val = int(default)
    if min_val is not None and val < min_val:
        val = min_val
    if max_val is not None and val > max_val:
        val = max_val
    return val


def _sanitize_mojibake_text(text: str) -> str:
    """
    Best-effort fixer for common UTF-8 mojibake sequences that frequently
    appear in committee outputs (especially around apostrophes/quotes).

    This is intentionally conservative and only replaces a few known bad
    byte-sequence patterns with their intended ASCII equivalents.
    """
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
    """
    Heuristic to detect essentially-empty Song Graph candidates.

    Used only when the expected_schema is a {"song": SONG_GRAPH_SCHEMA} wrapper.
    """
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
    # Treat as empty when there is no usable timing info and no structural/voice/instrument/motif content.
    if (not isinstance(bpm, (int, float)) or float(bpm) == 0.0) and not sections and not voices and not instruments and not motifs and not lyrics_sections:
        return True
    return False


# Committee model routing and context configuration.
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

# Use orchestrator-wide logging; no per-module basicConfig. All committee logs
# go through the same handlers configured in app.main.
log = logging.getLogger("orchestrator.committee")


async def call_ollama(base_url: str, payload: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    """
    Low-level Ollama generate call used by committee debate.

    This function is owned by the committee layer so that all committee-related
    backend calls are centralized here.

    IMPORTANT: This function must **never** raise. All errors are logged and
    returned in a structured envelope so upstream planner/committee callers can
    surface them cleanly to HTTP responses instead of killing connections.
    """
    trace_key = str(trace_id or "committee")
    t_all = time.monotonic()
    model = payload.get("model")
    prompt = payload.get("prompt")
    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    log.info(
        "[committee] ollama.call.start trace_id=%s base=%s model=%s stream=%s options=%s prompt_chars=%d",
        trace_key,
        base_url,
        model,
        bool(payload.get("stream", False)),
        options,
        len(str(prompt or "")),
    )

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        # Log full prompt and model for complete visibility.
        log.info("[committee] ollama.request base=%s model=%s trace_id=%s prompt=%s", base_url, model, trace_key, prompt)

        # Trace request (keep it lean; logging carries the huge bits).
        emit_trace(
            STATE_DIR,
            trace_key,
            "committee.ollama.request",
            {
                "trace_id": trace_key,
                "base_url": base_url,
                "model": model,
                "options": options,
            },
        )

        resp: httpx.Response | None = None
        t_http = time.monotonic()
        try:
            resp = await client.post(f"{base_url.rstrip('/')}/api/generate", json=dict(payload))
        except httpx.ConnectError as exc:
            msg = str(exc)
            log.error(
                "[committee] ollama.connect_error trace_id=%s base=%s model=%s dur_ms=%d error=%s",
                trace_key,
                base_url,
                model,
                int((time.monotonic() - t_http) * 1000.0),
                msg,
            )
            emit_trace(
                STATE_DIR,
                trace_key,
                "committee.ollama.connect_error",
                {"trace_id": trace_key, "base_url": base_url, "error": msg},
            )
            return {"ok": False, "error": {"code": "ollama_connect_error", "message": msg, "base_url": base_url}}
        except Exception as exc:  # pragma: no cover - defensive logging
            msg = str(exc)
            log.error(
                "[committee] ollama.request_error trace_id=%s base=%s model=%s dur_ms=%d error=%s",
                trace_key,
                base_url,
                model,
                int((time.monotonic() - t_http) * 1000.0),
                msg,
                exc_info=True,
            )
            emit_trace(
                STATE_DIR,
                trace_key,
                "committee.ollama.request_error",
                {"trace_id": trace_key, "base_url": base_url, "error": msg},
            )
            return {"ok": False, "error": {"code": "ollama_request_error", "message": msg, "base_url": base_url}}

        raw_text = (resp.text or "") if resp is not None else ""
        status_code = int(getattr(resp, "status_code", 0) or 0) if resp is not None else 0
        headers = dict(resp.headers) if resp is not None else {}

        log.info(
            "[committee] ollama.http.response trace_id=%s base=%s model=%s status=%d http_dur_ms=%d raw_chars=%d content_type=%s",
            trace_key,
            base_url,
            model,
            status_code,
            int((time.monotonic() - t_http) * 1000.0),
            len(raw_text),
            headers.get("content-type"),
        )
        log.info(
            "[committee] ollama.http.raw trace_id=%s base=%s model=%s status=%d raw=%s",
            trace_key,
            base_url,
            model,
            status_code,
            raw_text,
        )

        parser = JSONParser()
        t_parse = time.monotonic()
        try:
            parsed_obj = parser.parse(raw_text or "{}", {})
            parsed = parsed_obj if isinstance(parsed_obj, dict) else {}
        except Exception as exc:  # pragma: no cover - defensive logging
            msg = str(exc)
            log.error(
                "[committee] ollama.response_parse_error trace_id=%s base=%s model=%s status=%d parse_dur_ms=%d error=%s raw=%s",
                trace_key,
                base_url,
                model,
                status_code,
                int((time.monotonic() - t_parse) * 1000.0),
                msg,
                raw_text,
                exc_info=True,
            )
            emit_trace(
                STATE_DIR,
                trace_key,
                "committee.ollama.response_parse_error",
                {"trace_id": trace_key, "base_url": base_url, "status_code": status_code, "error": msg, "raw": raw_text},
            )
            return {
                "ok": False,
                "error": {"code": "ollama_bad_json", "message": msg, "status": status_code, "base_url": base_url},
            }

        if parser.errors:
            log.warning(
                "[committee] ollama.parser_errors trace_id=%s base=%s model=%s status=%d last_error=%s errors=%s",
                trace_key,
                base_url,
                model,
                status_code,
                parser.last_error,
                list(parser.errors)[:25],
            )
        log.info(
            "[committee] ollama.parsed trace_id=%s base=%s model=%s status=%d parse_dur_ms=%d keys=%s",
            trace_key,
            base_url,
            model,
            status_code,
            int((time.monotonic() - t_parse) * 1000.0),
            sorted(list(parsed.keys())),
        )

        if not (200 <= status_code < 300):
            emit_trace(
                STATE_DIR,
                trace_key,
                "committee.ollama.http_error",
                {"trace_id": trace_key, "base_url": base_url, "status_code": status_code, "body": parsed},
            )
            log.error(
                "[committee] ollama.http_error trace_id=%s base=%s model=%s status=%d dur_ms=%d body=%s",
                trace_key,
                base_url,
                model,
                status_code,
                int((time.monotonic() - t_all) * 1000.0),
                parsed,
            )
            return {
                "ok": False,
                "error": {
                    "code": "ollama_http_error",
                    "message": f"ollama returned HTTP {status_code}",
                    "status": status_code,
                    "base_url": base_url,
                    "details": {"body": parsed},
                },
            }

        response_str = parsed.get("response") if isinstance(parsed.get("response"), str) else ""
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
            "[committee] ollama.response base=%s model=%s trace_id=%s status=%s response=%s usage=%s",
            base_url,
            model,
            trace_key,
            status_code,
            response_str,
            usage,
        )
        log.info(
            "[committee] ollama.call.finish trace_id=%s ok=true dur_ms=%d status=%d response_chars=%d usage=%s",
            trace_key,
            int((time.monotonic() - t_all) * 1000.0),
            status_code,
            len(response_str or ""),
            usage,
        )

        emit_trace(
            STATE_DIR,
            trace_key,
            "committee.ollama.response",
            {"trace_id": trace_key, "status_code": status_code, "usage": usage or {}, "response": data},
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
    log.info(
        "[committee] ollama.payload.rendered model=%s num_ctx=%s temperature=%s messages=%d prompt_chars=%d dur_ms=%d",
        str(model),
        int(num_ctx),
        float(temperature),
        len(messages or []),
        len(prompt),
        int((time.monotonic() - t0) * 1000.0),
    )
    # ICW / context orchestration: keep the rendered prompt within a rough
    # character budget derived from DEFAULT_NUM_CTX so we don't saturate the
    # backend's context window and get garbage like ".TabStop" instead of real
    # output. No try/except here; DEFAULT_NUM_CTX is validated at import time.
    max_chars = DEFAULT_NUM_CTX * 4  # ≈4 chars/token heuristic
    if len(prompt) > max_chars:
        # Prefer to retain the tail of the prompt (latest user + tool context)
        # while trimming older system/history content from the front.
        trimmed = prompt[-max_chars:]
        log.info(
            "[committee] prompt.truncated model=%s num_ctx=%s orig_chars=%d kept_chars=%d",
            model,
            DEFAULT_NUM_CTX,
            len(prompt),
            len(trimmed),
        )
        log.info(
            "[committee] prompt.truncated.details model=%s default_num_ctx=%d orig_chars=%d kept_chars=%d max_chars=%d",
            str(model),
            int(DEFAULT_NUM_CTX),
            len(prompt),
            len(trimmed),
            int(max_chars),
        )
        prompt = trimmed
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Ollama expects keep_alive at the top-level (not inside options).
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
    """
    Internal-only primitive: call exactly one committee participant.
    """
    t0 = time.monotonic()
    cfg = _participant_for(member_id)
    if not cfg:
        log.error(
            "[committee] member.finish trace_id=%s member=%s ok=false dur_ms=%d error=unknown_member",
            str(trace_id or "committee"),
            str(member_id or ""),
            int((time.monotonic() - t0) * 1000.0),
        )
        return {"ok": False, "error": {"code": "unknown_member", "message": f"unknown committee member: {member_id}"}}
    base = (cfg.get("base") or "").rstrip("/") or QWEN_BASE_URL
    model = cfg.get("model") or QWEN_MODEL_ID
    log.info(
        "[committee] member.start trace_id=%s member=%s base=%s model=%s messages=%d temperature=%s",
        str(trace_id or "committee"),
        str(member_id or ""),
        str(base),
        str(model),
        len(messages or []),
        float(temperature),
    )
    payload = build_ollama_payload(messages or [], model, DEFAULT_NUM_CTX, temperature)
    res = await call_ollama(base, payload, trace_id=trace_id)
    if isinstance(res, dict):
        res["_base_url"] = base
        res["_model"] = model
        res["_member"] = str(member_id or "")
        log.info(
            "[committee] member.finish trace_id=%s member=%s ok=%s dur_ms=%d response_chars=%d has_error=%s",
            str(trace_id or "committee"),
            str(member_id or ""),
            bool(res.get("ok", True)),
            int((time.monotonic() - t0) * 1000.0),
            len(str(res.get("response") or "")),
            bool(res.get("error")),
        )
        return res
    log.error(
        "[committee] member.finish trace_id=%s member=%s ok=false dur_ms=%d error=invalid_member_response resp=%r",
        str(trace_id or "committee"),
        str(member_id or ""),
        int((time.monotonic() - t0) * 1000.0),
        res,
    )
    return {"ok": False, "error": {"code": "invalid_member_response", "message": str(res)}}


async def committee_synth_text(
    messages: List[Dict[str, Any]],
    *,
    trace_id: str,
    temperature: float = 0.0,
    synth_member: str = "qwen",
) -> Dict[str, Any]:
    """
    Internal-only primitive: deterministic synthesis call (single member).
    """
    log.info(
        "[committee] synth.start trace_id=%s synth_member=%s messages=%d temperature=%s",
        str(trace_id or "committee"),
        str(synth_member or "qwen"),
        len(messages or []),
        float(temperature),
    )
    return await committee_member_text(synth_member, messages, trace_id=trace_id, temperature=temperature)


async def committee_ai_text(
    messages: List[Dict[str, Any]],
    trace_id: str,
    rounds: int | None = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Core committee primitive:
    - Draft answers per member
    - Cross-critique per member
    - Revision per member
    - Deterministic synthesis (single member) into result.text

    IMPORTANT:
    - Only the committee layer may call Ollama.
    - Callers MUST NOT implement cross-debate by calling this function multiple
      times pretending it's "per member". Per-member calls are internal helpers.
    """
    trace_key = str(trace_id or "committee")
    effective_rounds = max(1, int(rounds if rounds is not None else DEFAULT_COMMITTEE_ROUNDS))
    member_ids = [p.get("id") or "member" for p in COMMITTEE_PARTICIPANTS]

    answers: Dict[str, str] = {}
    critiques: Dict[str, str] = {}
    member_results: Dict[str, Dict[str, Any]] = {}
    critique_results: Dict[str, Dict[str, Any]] = {}

    t0 = time.monotonic()
    log.info(
        "[committee] run.start trace_id=%s rounds=%d temperature=%s participants=%s messages=%d",
        trace_key,
        int(effective_rounds),
        float(temperature),
        member_ids,
        len(messages or []),
    )
    emit_trace(
        STATE_DIR,
        trace_key,
        "committee.start",
        {
            "trace_id": trace_key,
            "rounds": int(effective_rounds),
            "temperature": float(temperature),
            "participants": member_ids,
            "messages": messages or [],
        },
    )

    for r in range(effective_rounds):
        # Phase 1: Draft answers
        log.info("[committee] round.start trace_id=%s round=%d", trace_key, int(r + 1))
        for mid in member_ids:
            log.info("[committee] member_draft.start trace_id=%s round=%d member=%s", trace_key, int(r + 1), mid)
            draft_lines = [
                f"You are committee member {mid}. Provide your best answer to the user.",
                "All content must be written in English only. Do NOT respond in any other language.",
            ]
            member_msgs = list(messages or []) + [{"role": "system", "content": "\n\n".join(draft_lines)}]
            emit_trace(
                STATE_DIR,
                trace_key,
                "committee.member_draft",
                {"trace_id": trace_key, "round": int(r + 1), "member": mid},
            )
            res = await committee_member_text(mid, member_msgs, trace_id=trace_key, temperature=temperature)
            member_results[mid] = res if isinstance(res, dict) else {}
            txt = ""
            if isinstance(res, dict) and res.get("ok") is not False:
                txt = str(res.get("response") or "").strip()
            answers[mid] = txt
            log.info(
                "[committee] member_draft.finish trace_id=%s round=%d member=%s ok=%s answer_chars=%d has_error=%s",
                trace_key,
                int(r + 1),
                mid,
                bool(isinstance(res, dict) and res.get("ok", True)),
                len(txt),
                bool(isinstance(res, dict) and res.get("error")),
            )

        # Phase 2: Cross-critique
        log.info(
            "[committee] round.critique.start trace_id=%s round=%d answers_chars=%s",
            trace_key,
            int(r + 1),
            {mid: len(answers.get(mid) or "") for mid in member_ids},
        )
        critiques = {}
        for mid in member_ids:
            log.info("[committee] member_critique.start trace_id=%s round=%d member=%s", trace_key, int(r + 1), mid)
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
                trace_key,
                "committee.member_critique",
                {"trace_id": trace_key, "round": int(r + 1), "member": mid},
            )
            res = await committee_member_text(mid, member_msgs, trace_id=trace_key, temperature=temperature)
            critique_results[mid] = res if isinstance(res, dict) else {}
            crit_txt = ""
            if isinstance(res, dict) and res.get("ok") is not False:
                crit_txt = str(res.get("response") or "").strip()
            critiques[mid] = crit_txt
            log.info(
                "[committee] member_critique.finish trace_id=%s round=%d member=%s ok=%s critique_chars=%d has_error=%s others_count=%d",
                trace_key,
                int(r + 1),
                mid,
                bool(isinstance(res, dict) and res.get("ok", True)),
                len(crit_txt),
                bool(isinstance(res, dict) and res.get("error")),
                sum(1 for oid in member_ids if oid != mid and (answers.get(oid) or "").strip()),
            )

        # Phase 3: Revision
        log.info(
            "[committee] round.revision.start trace_id=%s round=%d critiques_chars=%s",
            trace_key,
            int(r + 1),
            {mid: len(critiques.get(mid) or "") for mid in member_ids},
        )
        for mid in member_ids:
            log.info("[committee] member_revision.start trace_id=%s round=%d member=%s", trace_key, int(r + 1), mid)
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
                trace_key,
                "committee.member_revision",
                {"trace_id": trace_key, "round": int(r + 1), "member": mid},
            )
            res = await committee_member_text(mid, member_msgs, trace_id=trace_key, temperature=temperature)
            member_results[mid] = res if isinstance(res, dict) else {}
            txt = ""
            if isinstance(res, dict) and res.get("ok") is not False:
                txt = str(res.get("response") or "").strip()
            answers[mid] = txt
            log.info(
                "[committee] member_revision.finish trace_id=%s round=%d member=%s ok=%s answer_chars=%d has_error=%s",
                trace_key,
                int(r + 1),
                mid,
                bool(isinstance(res, dict) and res.get("ok", True)),
                len(txt),
                bool(isinstance(res, dict) and res.get("error")),
            )

    member_summaries: List[Dict[str, Any]] = [{"member": mid, "answer": (answers.get(mid) or "")} for mid in member_ids]
    critique_summaries: List[Dict[str, Any]] = [{"member": mid, "critique": (critiques.get(mid) or "")} for mid in member_ids]

    # Deterministic synthesis (single member)
    log.info(
        "[committee] synth.build trace_id=%s answers_chars=%s critiques_present=%s",
        trace_key,
        {mid: len(answers.get(mid) or "") for mid in member_ids},
        bool(any((critiques.get(mid) or "").strip() for mid in member_ids)),
    )
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
        trace_key,
        "committee.synth.request",
        {"trace_id": trace_key, "synth_member": "qwen"},
    )
    t_synth = time.monotonic()
    synth_env = await committee_synth_text(synth_messages, trace_id=f"{trace_key}.synth", temperature=0.0, synth_member="qwen")
    synth_text = ""
    if isinstance(synth_env, dict) and synth_env.get("ok") is not False:
        synth_text = str(synth_env.get("response") or "").strip()
    log.info(
        "[committee] synth.finish trace_id=%s ok=%s dur_ms=%d synth_chars=%d has_error=%s",
        trace_key,
        bool(isinstance(synth_env, dict) and synth_env.get("ok", True)),
        int((time.monotonic() - t_synth) * 1000.0),
        len(synth_text),
        bool(isinstance(synth_env, dict) and synth_env.get("error")),
    )

    # Fallback if synth is empty: choose best member answer deterministically.
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
                "[committee] synth.fallback trace_id=%s chosen_member=%s chosen_chars=%d",
                trace_key,
                str(best_mid),
                len(final_text),
            )

    ok = bool(isinstance(final_text, str) and final_text.strip())

    backend_errors: List[Dict[str, Any]] = []
    for mid, mres in member_results.items():
        if isinstance(mres, dict) and mres.get("ok") is False and isinstance(mres.get("error"), dict):
            backend_errors.append({"member": mid, "error": mres.get("error")})
    if isinstance(synth_env, dict) and synth_env.get("ok") is False and isinstance(synth_env.get("error"), dict):
        backend_errors.append({"member": "synth", "error": synth_env.get("error")})

    emit_trace(
        STATE_DIR,
        trace_key,
        "committee.finish",
        {"trace_id": trace_key, "ok": ok, "final_text": final_text, "members": member_summaries, "critiques": critique_summaries},
    )
    log.info(
        "[committee] run.finish trace_id=%s ok=%s dur_ms=%d final_chars=%d backend_errors=%s",
        trace_key,
        ok,
        int((time.monotonic() - t0) * 1000.0),
        len(final_text or ""),
        backend_errors,
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
        log.error(
            "[committee] no_answer trace_id=%s backend_errors=%s members=%s",
            trace_key,
            backend_errors,
            members_status,
        )

    result_payload: Dict[str, Any] = {
        "text": final_text,
        "members": member_summaries,
        "critiques": critique_summaries,
        "qwen": member_results.get("qwen") or {},
        "glm": member_results.get("glm") or {},
        "deepseek": member_results.get("deepseek") or {},
        "synth": (dict(synth_env) if isinstance(synth_env, dict) else {"response": final_text}),
        "critique_envelopes": critique_results,
    }

    return {
        "schema_version": 1,
        "trace_id": trace_id or "committee",
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
    """
    Build a literal JSON skeleton from an expected_schema that may contain
    Python type objects (str, int, float, dict, list) as leaves.

    This is ONLY for prompting the JSONFixer model; the original expected_schema
    (with type objects) is still used for JSONParser coercion/validation.
    """
    # Structured containers: recurse into dicts/lists
    if isinstance(expected, dict):
        return {k: _schema_to_template(v) for k, v in expected.items()}
    if isinstance(expected, list):
        # For list schemas, use the first element as the prototype when present.
        if not expected:
            return []
        return [_schema_to_template(expected[0])]
    # Type objects: map Python types to concrete JSON-compatible sample values.
    # This is the core path that prevents 'type' objects from leaking into
    # json.dumps calls when we render schema skeletons into prompts.
    if isinstance(expected, type):
        # Booleans first (bool is a subclass of int)
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
        # Unknown type object: fall back to null in JSON
        return None
    # Primitive instances or anything else that is already JSON-serializable
    # are returned as-is; non-serializable objects will be caught by json.dumps
    # at call sites if they appear here.
    return expected


def _default_for(expected: Any) -> Any:
    """
    Return a JSON-serializable default value for a given expected schema type.
    """
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
    """
    Treat "empty" scalar values (0, 0.0, "", None or blank strings) as defaults
    that should not overwrite richer values from other candidates. Only when no
    candidate provides a non-default value do we fall back to these.
    """
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
    """
    Use the committee to coerce a messy text blob into strict JSON matching expected_schema.

    Fully traced so we can debug planner behavior:
      - committee.jsonify.start
      - committee.jsonify.committee_result
      - committee.jsonify.candidates
      - committee.jsonify.merge
    """
    trace_base = str(trace_id or "committee")
    trace_full = f"{trace_base}.jsonify"
    t0 = time.monotonic()
    log.info(
        "[committee.jsonify] start trace_id=%s trace_id_full=%s rounds=%d temperature=%s raw_chars=%d schema_type=%s",
        trace_base,
        trace_full,
        int(rounds if rounds is not None else DEFAULT_COMMITTEE_ROUNDS),
        float(temperature),
        len(raw_text or ""),
        str(type(expected_schema).__name__),
    )
    schema_template = _schema_to_template(expected_schema)
    schema_desc = json.dumps(schema_template, ensure_ascii=False)
    raw_preview = (raw_text or "")[:600] if isinstance(raw_text, str) else ""
    emit_trace(
        STATE_DIR,
        trace_base,
        "committee.jsonify.start",
        {
            "trace_id": trace_base,
            "trace_id_full": trace_full,
            "schema_preview": schema_desc[:600],
            "schema_len": len(schema_desc),
            "raw_preview": raw_preview,
            "raw_len": len(raw_text or ""),
        },
    )
    # Log full raw_text and schema for debugging (no truncation).
    log.info("[committee.jsonify] start trace_id=%s schema=%s raw_text=%s", trace_base, schema_desc, raw_text)
    # Hard JSON contract: show the exact skeleton and forbid any deviation.
    # JSONFixer may receive arbitrary messy output (prose, partial JSON, mixed code).
    # It must always ignore refusals/prose and return a single valid JSON object.
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
        messages,
        trace_id=trace_full,
        rounds=rounds,
        temperature=temperature,
    )
    result_block = env.get("result") if isinstance(env, dict) else {}
    ok_env = bool(env.get("ok")) if isinstance(env, dict) else False
    txt_main_preview = ""
    if isinstance(result_block, dict):
        _txt = result_block.get("text")
        if isinstance(_txt, str):
            txt_main_preview = _txt[:600]
    emit_trace(
        STATE_DIR,
        trace_base,
        "committee.jsonify.committee_result",
        {
            "trace_id": trace_base,
            "trace_id_full": trace_full,
            "ok": ok_env,
            "main_text_preview": txt_main_preview,
        },
    )
    # Log full committee result text and envelope (no truncation).
    log.info(
        "[committee.jsonify] committee_result trace_id=%s ok=%s text=%s env=%s",
        trace_base,
        ok_env,
        result_block.get("text") if isinstance(result_block, dict) else None,
        result_block,
    )

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

    emit_trace(
        STATE_DIR,
        trace_base,
        "committee.jsonify.candidates",
        {
            "trace_id": trace_base,
            "count": len(candidates),
            "first_preview": (candidates[0][:400] if candidates else ""),
        },
    )
    # Log all candidate texts for JSON parsing.
    log.info(
        "[committee.jsonify] candidates trace_id=%s count=%d candidates=%s",
        trace_base,
        len(candidates),
        candidates,
    )

    parsed_candidates: List[Dict[str, Any]] = []

    # Detect Song Graph wrapper schemas so we can handle parse failures with a
    # canonical empty song object and downweight empty candidates.
    song_schema: Dict[str, Any] | None = None
    if isinstance(expected_schema, dict):
        maybe_song = expected_schema.get("song")
        if isinstance(maybe_song, dict) and "global" in maybe_song and "sections" in maybe_song:
            song_schema = maybe_song

    for idx, txt in enumerate(candidates):
        log.info(
            "[committee.jsonify] parse_candidate.start trace_id=%s index=%d text_chars=%d",
            trace_base,
            int(idx),
            len(txt or ""),
        )
        log.info(
            "[committee.jsonify] parse_candidate trace_id=%s index=%d text=%s",
            trace_base,
            idx,
            txt,
        )
        clean_txt = _sanitize_mojibake_text(txt or "")
        # Use a fresh parser per candidate so errors/repairs are per-candidate.
        parser = JSONParser()
        t_parse = time.monotonic()
        try:
            obj = parser.parse(clean_txt or "{}", expected_schema)
        except Exception as ex:  # pragma: no cover - defensive logging
            log.info(
                "[committee.jsonify] JSONParser exception trace_id=%s index=%d error=%s",
                trace_base,
                idx,
                str(ex),
            )
            # For Song Graph wrappers, treat parse failures as an explicit
            # empty song shell instead of dropping the candidate entirely.
            if song_schema is not None:
                try:
                    obj = parser.parse("{}", expected_schema)
                except Exception as ex2:
                    log.warning(
                        "[committee.jsonify] JSONParser exception on fallback trace_id=%s index=%d error=%s",
                        trace_base,
                        idx,
                        str(ex2),
                        exc_info=True,
                    )
                    log.error(
                        "[committee.jsonify] parse_candidate.finish trace_id=%s index=%d ok=false dur_ms=%d",
                        trace_base,
                        int(idx),
                        int((time.monotonic() - t_parse) * 1000.0),
                        exc_info=True,
                    )
                    continue
            else:
                log.error(
                    "[committee.jsonify] parse_candidate.finish trace_id=%s index=%d ok=false dur_ms=%d",
                    trace_base,
                    int(idx),
                    int((time.monotonic() - t_parse) * 1000.0),
                    exc_info=True,
                )
                continue

        # Surface JSONParser-internal errors (which never raise) so pack/compression
        # failures are visible in logs instead of being silently repaired.
        errors = list(parser.errors or [])
        last_error = parser.last_error
        if errors:
            log.warning(
                "[committee.jsonify] JSONParser reported errors trace_id=%s index=%d last_error=%s errors=%s",
                trace_base,
                idx,
                last_error,
                errors[:5],
            )
        log.info(
            "[committee.jsonify] parsed_candidate trace_id=%s index=%d obj=%s",
            trace_base,
            idx,
            obj,
        )
        if isinstance(obj, dict):
            parsed_candidates.append(obj)
            log.info(
                "[committee.jsonify] parse_candidate.finish trace_id=%s index=%d ok=true dur_ms=%d keys=%s",
                trace_base,
                int(idx),
                int((time.monotonic() - t_parse) * 1000.0),
                sorted(list(obj.keys())),
            )
        else:
            log.warning(
                "[committee.jsonify] parse_candidate.finish trace_id=%s index=%d ok=false dur_ms=%d non_dict_type=%s",
                trace_base,
                int(idx),
                int((time.monotonic() - t_parse) * 1000.0),
                str(type(obj).__name__),
            )

    # Merge candidates field-wise, preferring the first non-empty value across candidates.
    #
    # CRITICAL: do not drop "extra" keys the model returned. Callers (especially
    # tool pipelines) may rely on additional fields outside expected_schema.
    # We therefore build merged as a *superset union* of parsed candidates and
    # then apply schema-key selection logic on top.
    merged: Any = {}

    # For Song Graph wrappers, drop obviously-empty/default song candidates from
    # consideration when at least one non-empty candidate exists, and reorder
    # parsed_candidates so the richest candidate is considered first.
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
            # Sort by: most sections, then most (voices+instruments+motifs), then lowest index.
            rich_sorted = sorted(rich, key=lambda t: (-t[1], -t[2], t[0]))
            best_idx = rich_sorted[0][0]
            log.info(
                "[committee.jsonify] song_richness trace_id=%s chosen_index=%d top=%s",
                trace_base,
                int(best_idx),
                [{"idx": i, "sections": s, "richness": r} for (i, s, r) in rich_sorted[:10]],
            )
            reordered: List[Dict[str, Any]] = []
            best_cand = parsed_candidates[best_idx]
            reordered.append(best_cand)
            for i, cand in enumerate(parsed_candidates):
                if i == best_idx:
                    continue
                song_obj = cand.get("song")
                # Skip pure-default/empty song shells when we have at least one rich candidate.
                if isinstance(song_obj, dict) and not _is_empty_song_candidate(song_obj):
                    reordered.append(cand)
            parsed_candidates = reordered

    if isinstance(expected_schema, dict):
        # 1) Preserve all non-schema keys across candidates (top-level superset union).
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

        # 2) For schema keys, run the existing "best value wins" logic.
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
                    # Special-case Song Graph wrapper: prefer structurally
                    # non-empty song candidates (with bpm/sections/lyrics)
                    # over default/empty shells.
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
                    # Scalars: only accept non-default values; 0/0.0/""/None are
                    # treated as weaker than any non-default from another
                    # candidate and used only as a final fallback.
                    if isinstance(v, expected_type) and not _is_default_scalar(v, expected_type):
                        merged[key] = v
                        value_set = True
                        break
            if not value_set:
                merged[key] = _default_for(expected_type)
            log.info(
                "[committee.jsonify] merge.key trace_id=%s key=%s set=%s expected=%s chosen_type=%s",
                trace_base,
                str(key),
                bool(value_set),
                str(expected_type),
                str(type(merged.get(key)).__name__) if isinstance(merged, dict) else None,
            )
    else:
        # Non-dict schema: just return the first successful parse or a default.
        if parsed_candidates:
            merged = parsed_candidates[0]
        else:
            # No successful candidates: still return a value shaped to the expected schema.
            # (The previous nested isinstance(expected_schema, dict) check was unreachable here.)
            if isinstance(expected_schema, list):
                seed: Any = []
            else:
                seed = _default_for(expected_schema)
            try:
                merged = JSONParser().parse(seed, expected_schema)
            except Exception as exc:
                # Defensive: JSONParser should not raise here, but if it does, log so we don't silently
                # degrade the output shape (this can otherwise look like "the request died").
                log.warning(
                    "[committee.jsonify] merge.seed_parse_failed trace_id=%s schema_type=%s seed_type=%s error=%s",
                    trace_base,
                    str(type(expected_schema).__name__),
                    str(type(seed).__name__),
                    str(exc),
                    exc_info=True,
                )
                merged = seed

    emit_trace(
        STATE_DIR,
        trace_base,
        "committee.jsonify.merge",
        {
            "trace_id": trace_base,
            "trace_id_full": trace_full,
            "candidates": len(candidates),
            "parsed_candidates": len(parsed_candidates),
            "merged_keys": sorted(list(merged.keys())) if isinstance(merged, dict) else [],
        },
    )
    log.info(
        "[committee.jsonify] merge trace_id=%s candidates=%d parsed=%d keys=%s",
        trace_base,
        len(candidates),
        len(parsed_candidates),
        ",".join(sorted(list(merged.keys()))) if isinstance(merged, dict) else "",
    )
    log.info(
        "[committee.jsonify] finish trace_id=%s trace_id_full=%s dur_ms=%d candidates=%d parsed_candidates=%d merged_is_dict=%s merged_keys=%s",
        trace_base,
        trace_full,
        int((time.monotonic() - t0) * 1000.0),
        int(len(candidates)),
        int(len(parsed_candidates)),
        bool(isinstance(merged, dict)),
        (sorted(list(merged.keys())) if isinstance(merged, dict) else []),
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
