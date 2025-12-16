from __future__ import annotations

import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any
import hashlib

from .tokenizer import bytes_len
from .tokenizer import byte_budget_for_model
from .compress import (
    cleanup_lossless,
    compact_numbers_tables,
    summarize_block_extractive,
    facts_pocket,
    attach_pocket,
)
from .relevance import score_chunk
from .continuation import make_system_hint
from .continuation import state_hash_from
from ..rag.hygiene import rag_filter, evidence_binding_footer
from ..artifacts.shard import newest_part, list_parts

log = logging.getLogger("orchestrator.icw.assembler")


def render_header(entity_header: str) -> str:
    return entity_header


def render_anchor(anchor: str) -> str:
    return anchor


def render_chunk(chunk: str) -> str:
    return chunk


def make_entity_header(state) -> str:
    # Build compact header with entities/goals/constraints (deterministic order)
    if not isinstance(state, dict):
        log.warning("icw.assembler make_entity_header state not dict type=%s", type(state).__name__)
        state = {}
    ents = state.get("entities", [])
    if not isinstance(ents, list):
        ents = [ents]
    goal = str(state.get("goals", "") or "")
    safe_ents: list[str] = []
    for e in ents:
        if not e:
            continue
        try:
            safe_ents.append(str(e))
        except Exception as exc:
            # Defensive: avoid silently dropping weird entity objects.
            log.warning("icw.assembler entity str() failed type=%s: %s", type(e).__name__, exc, exc_info=True)
            safe_ents.append(repr(e))
    return f"[Goal] {goal}\n[Entities] " + ", ".join(sorted([e for e in safe_ents if e]))[:400]


def _joined(parts: list[str]) -> str:
    return "\n\n".join([p for p in parts if isinstance(p, str) and p])


def compress_until_fit(parts: list[str], budget_bytes: int) -> list[str]:
    t0 = time.perf_counter()
    try:
        b = int(budget_bytes)
    except Exception as exc:  # pragma: no cover
        log.warning("icw.assembler compress_until_fit bad budget_bytes=%r: %s", budget_bytes, exc, exc_info=True)
        b = 0
    if b <= 0:
        b = 4096
    # Pass 0: lossless cleanup
    safe_parts: list[str] = []
    coerced = 0
    for p in (parts or []):
        if isinstance(p, str):
            safe_parts.append(p)
        elif p is None:
            safe_parts.append("")
            coerced += 1
        else:
            safe_parts.append(str(p))
            coerced += 1
    if coerced:
        log.warning("icw.assembler compress_until_fit coerced_non_str_parts=%s", coerced)
    parts = [cleanup_lossless(p) for p in safe_parts]
    # Pass 1: number/table compaction
    parts = [compact_numbers_tables(p) for p in parts]
    # Pass 2: summarize lowest-priority items (end of list first)
    i = len(parts) - 1
    iters = 0
    before = bytes_len(_joined(parts))
    while bytes_len(_joined(parts)) > b and i > 2:
        iters += 1
        p = parts[i]
        pocket = facts_pocket(p)
        parts[i] = attach_pocket(summarize_block_extractive(p, 0.5), pocket)
        i -= 1
        if i < 3:
            break
    after = bytes_len(_joined(parts))
    dt_ms = int((time.perf_counter() - t0) * 1000)
    if before > b:
        log.info(
            "icw.assembler compress_until_fit budget_bytes=%s before=%s after=%s iters=%s ms=%s",
            b,
            before,
            after,
            iters,
            dt_ms,
        )
    return parts


def _normalize_candidate(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        try:
            return json.dumps(c, ensure_ascii=False)
        except Exception as exc:
            log.error("icw.assembler candidate json.dumps failed: %s", exc, exc_info=True)
            return repr(c)
    if c is None:
        return ""
    try:
        return str(c)
    except Exception as exc:  # pragma: no cover
        log.error("icw.assembler candidate str() failed type=%s: %s", type(c), exc, exc_info=True)
        return repr(c)


def assemble_window(request_msg: dict, state: dict, in_limit_bytes: int, step_out_tokens: int):
    """Return a prompt string that fits the byte budget."""
    t0 = time.perf_counter()
    if not isinstance(request_msg, dict):
        log.error("icw.assembler assemble_window request_msg not dict type=%s", type(request_msg).__name__)
        request_msg = {}
    if not isinstance(state, dict):
        log.error("icw.assembler assemble_window state not dict type=%s", type(state).__name__)
        state = {}
    try:
        budget_bytes = int(in_limit_bytes)
    except Exception as exc:
        log.warning("icw.assembler assemble_window bad in_limit_bytes=%r: %s", in_limit_bytes, exc, exc_info=True)
        budget_bytes = 0
    if budget_bytes <= 0:
        budget_bytes = 4096
    try:
        out_tokens = int(step_out_tokens)
    except Exception as exc:
        log.warning("icw.assembler assemble_window bad step_out_tokens=%r: %s", step_out_tokens, exc, exc_info=True)
        out_tokens = 0
    goal = str(request_msg.get("content", "") or "")
    anchor = str(state.get("anchor_text", "") or "")  # recent exact turns (verbatim)
    header = make_entity_header({"entities": state.get("entities", []), "goals": goal})
    # Candidates = history chunks + RAG chunks + artifacts summaries (strings)
    raw_candidates = state.get("candidates", []) or []
    if not isinstance(raw_candidates, list):
        log.warning("icw.assembler candidates not list type=%s", type(raw_candidates).__name__)
        raw_candidates = [raw_candidates]
    candidates = [_normalize_candidate(c) for c in raw_candidates if c is not None]
    # Optional: append filtered RAG evidence footer at the tail (newest-first)
    retrieved = state.get("retrieved") or state.get("retrieved_chunks") or []
    footer = ""
    try:
        if not isinstance(retrieved, list):
            log.warning("icw.assembler retrieved not list type=%s", type(retrieved).__name__)
            retrieved = [retrieved]
        if retrieved:
            ttl = None
            try:
                ttl = int(os.getenv("RAG_TTL_SECONDS", "3600"))
            except Exception as exc:  # defensive logging
                log.error("icw.assembler RAG_TTL_SECONDS parse failed: %s", exc, exc_info=True)
                ttl = None
            distilled = rag_filter(retrieved, ttl_s=ttl)
            footer = evidence_binding_footer(distilled)
    except Exception as exc:  # defensive logging
        log.error("icw.assembler RAG footer build failed: %s", exc, exc_info=True)
        footer = "[RAG] evidence temporarily unavailable due to error."

    # Rank by relevance
    ranked = sorted(candidates, key=lambda t: score_chunk(t, goal), reverse=True)
    # Build parts in priority order
    system_hint = make_system_hint(out_tokens, state.get("state_hash", ""))
    parts: list[str] = [system_hint, render_anchor(anchor), render_header(header)]
    # Artifacts preference (research ledger): include newest shard summary if present (best-effort)
    cid = state.get("cid") if isinstance(state, dict) else None
    try:
        if cid:
            root = os.path.join(os.getenv("UPLOAD_DIR", "/workspace/uploads"), "artifacts", "research", str(cid))
            latest = newest_part(root, "ledger")
            if latest:
                _bytes_raw = latest.get("bytes") if isinstance(latest, dict) else None
                try:
                    _bytes_int = int(_bytes_raw or 0)
                except Exception as exc:
                    log.warning("icw.assembler bad ledger bytes=%r; defaulting to 0", _bytes_raw, exc_info=True)
                    _bytes_int = 0
                parts.append(render_chunk(f"[Ledger newest] {latest.get('path')} ~{_bytes_int} bytes"))
                # Summarize older shards as one-liners
                older = list_parts(root, "ledger")[:-1]
                for p in older[-5:]:  # cap summaries
                    sha = (p.get("sha256") or "")
                    parts.append(
                        render_chunk(f"[Ledger older] {p.get('path')} sha256:{sha[:8]}")
                    )
    except Exception as exc:  # defensive logging
        log.error("icw.assembler ledger summary build failed for cid=%r: %s", cid, exc, exc_info=True)

    log.info(
        "icw.assembler assemble_window start cid=%r budget_bytes=%s goal_len=%s anchor_len=%s candidates=%s retrieved=%s",
        cid,
        budget_bytes,
        len(goal),
        len(anchor),
        len(ranked),
        len(retrieved) if isinstance(retrieved, list) else 0,
    )
    for c in ranked:
        parts.append(render_chunk(c))
        prompt = _joined(parts)
        if bytes_len(prompt) > budget_bytes:
            parts = compress_until_fit(parts, budget_bytes)
            prompt = _joined(parts)
            if bytes_len(prompt) > budget_bytes:
                parts.pop()  # defer this chunk to next window
                break
    if footer:
        parts.append(footer)
    final_prompt = _joined(parts)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "icw.assembler assemble_window done bytes=%s budget=%s parts=%s ms=%s",
        bytes_len(final_prompt),
        budget_bytes,
        len(parts),
        dt_ms,
    )
    return SimpleNamespace(prompt=final_prompt, target_output_tokens=out_tokens)


def pack_icw_system_frame_from_messages(
    messages: list[dict],
    *,
    cid: str | None,
    goals: str,
    model_ctx_limit_tokens: int,
    pct_budget: float = 0.65,
    step_out_tokens: int = 512,
) -> tuple[dict | None, str | None, dict]:
    """
    Build a single ICW system frame for downstream prompts.

    This is the one-call ICW interface: callers pass message history and a goal,
    and ICW returns a compact system frame that fits a stable byte budget.

    Returns: (system_frame_or_none, pack_hash_or_none, meta)
    """
    t0 = time.perf_counter()
    last_user = str(goals or "").strip()
    if not last_user:
        log.warning("icw.assembler pack empty_goals cid=%r", cid)
        return None, None, {"ok": True, "reason": "empty_goals"}

    msgs = messages or []
    if not isinstance(msgs, list):
        log.warning("icw.assembler pack messages not list type=%s", type(msgs).__name__)
        msgs = [msgs]  # type: ignore[list-item]
    # Budget: token limit -> byte budget, then allocate pct_budget for ICW.
    try:
        _ctx_limit = int(model_ctx_limit_tokens)
    except Exception as exc:
        log.warning("icw.assembler pack bad model_ctx_limit_tokens=%r: %s", model_ctx_limit_tokens, exc, exc_info=True)
        _ctx_limit = 2048
    if _ctx_limit <= 0:
        _ctx_limit = 2048
    try:
        total_budget_bytes = int(byte_budget_for_model(_ctx_limit))
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.assembler pack budget calc failed: %s", exc, exc_info=True)
        total_budget_bytes = 8192
    try:
        _pct = float(pct_budget)
    except Exception as exc:
        log.warning("icw.assembler pack bad pct_budget=%r: %s", pct_budget, exc, exc_info=True)
        _pct = 0.65
    if _pct <= 0.0 or _pct > 1.0:
        _pct = 0.65
    try:
        icw_budget_bytes = int(total_budget_bytes * _pct)
    except Exception as exc:
        log.warning("icw.assembler pack budget multiply failed pct_budget=%r: %s", pct_budget, exc, exc_info=True)
        icw_budget_bytes = int(total_budget_bytes * 0.65)
    if icw_budget_bytes <= 0:
        icw_budget_bytes = max(1024, int(total_budget_bytes * 0.5))

    anchor_lines: list[str] = []
    for m in (msgs or [])[-8:]:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        content = str(m.get("content") or "").strip()
        if role and content:
            anchor_lines.append(f"{role}: {content}")
    anchor_text = "\n".join(anchor_lines)[:6000]

    candidates: list[str] = []
    for m in (msgs or []):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        content = str(m.get("content") or "").strip()
        if role and content:
            candidates.append(f"{role}: {content}")

    icw_state: dict = {
        "cid": cid,
        "entities": [],
        "goals": last_user,
        "anchor_text": anchor_text,
        "candidates": candidates,
        "retrieved": [],
    }
    try:
        icw_state["state_hash"] = state_hash_from(icw_state)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.assembler pack state_hash_from failed cid=%r: %s", cid, exc, exc_info=True)
        return None, None, {"ok": False, "reason": "state_hash_failed"}

    req = {"role": "user", "content": last_user}
    # Defensive: step_out_tokens may be str-ish; default rather than failing the pack.
    _step_out = 512
    try:
        _step_out = int(step_out_tokens)
    except Exception as exc:
        log.warning("icw.assembler pack bad step_out_tokens=%r; defaulting to 512", step_out_tokens, exc_info=True)
        _step_out = 512
    if _step_out <= 0:
        _step_out = 512
    try:
        window = assemble_window(req, icw_state, int(icw_budget_bytes), int(_step_out))
    except Exception as exc:
        log.error("icw.assembler pack assemble_window failed cid=%r: %s", cid, exc, exc_info=True)
        return None, None, {"ok": False, "reason": "assemble_window_failed"}
    prompt = getattr(window, "prompt", "") if window is not None else ""
    prompt = str(prompt or "").strip()
    if not prompt:
        log.error("icw.assembler pack empty_prompt cid=%r budget_bytes=%s", cid, icw_budget_bytes)
        return None, None, {"ok": False, "reason": "empty_prompt"}

    pack_hash = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    frame = {"role": "system", "content": "### [ICW CONTEXT]\n" + prompt}
    # meta must never raise (pack is best-effort instrumentation).
    try:
        _pct_meta = float(_pct) if "_pct" in locals() else 0.65
    except Exception:
        _pct_meta = 0.65
    meta = {
        "ok": True,
        "budget_bytes": int(icw_budget_bytes),
        "total_budget_bytes": int(total_budget_bytes),
        "pct_budget": float(_pct_meta),
        "approx_chars": len(prompt),
        "state_hash": str(icw_state.get("state_hash") or ""),
    }
    dt_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "icw.assembler pack ok cid=%r msg_count=%s bytes=%s budget_bytes=%s ms=%s",
        cid,
        len(msgs),
        bytes_len(prompt),
        int(icw_budget_bytes),
        dt_ms,
    )
    return frame, pack_hash, meta

