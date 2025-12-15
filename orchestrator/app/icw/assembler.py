from __future__ import annotations

import json
import logging
import os
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
    ents = state.get("entities", [])
    goal = state.get("goals", "")
    return f"[Goal] {goal}\n[Entities] " + ", ".join(sorted([e for e in ents if e]))[:400]


def _joined(parts: list[str]) -> str:
    return "\n\n".join([p for p in parts if isinstance(p, str) and p])


def compress_until_fit(parts: list[str], budget_bytes: int) -> list[str]:
    # Pass 0: lossless cleanup
    parts = [cleanup_lossless(p) for p in parts]
    # Pass 1: number/table compaction
    parts = [compact_numbers_tables(p) for p in parts]
    # Pass 2: summarize lowest-priority items (end of list first)
    i = len(parts) - 1
    while bytes_len(_joined(parts)) > budget_bytes and i > 2:
        p = parts[i]
        pocket = facts_pocket(p)
        parts[i] = attach_pocket(summarize_block_extractive(p, 0.5), pocket)
        i -= 1
        if i < 3:
            break
    return parts


def _normalize_candidate(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        try:
            return json.dumps(c, ensure_ascii=False)
        except Exception:
            return repr(c)
    if c is None:
        return ""
    return str(c)


def assemble_window(request_msg: dict, state: dict, in_limit_bytes: int, step_out_tokens: int):
    """Return a prompt string that fits the byte budget."""
    goal = request_msg.get("content", "")
    anchor = state.get("anchor_text", "")  # recent exact turns (verbatim)
    header = make_entity_header({"entities": state.get("entities", []), "goals": goal})
    # Candidates = history chunks + RAG chunks + artifacts summaries (strings)
    raw_candidates = state.get("candidates", []) or []
    candidates = [_normalize_candidate(c) for c in raw_candidates if c is not None]
    # Optional: append filtered RAG evidence footer at the tail (newest-first)
    retrieved = state.get("retrieved") or state.get("retrieved_chunks") or []
    footer = ""
    try:
        if isinstance(retrieved, list) and retrieved:
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
    system_hint = make_system_hint(step_out_tokens, state.get("state_hash", ""))
    parts: list[str] = [system_hint, render_anchor(anchor), render_header(header)]
    # Artifacts preference (research ledger): include newest shard summary if present (best-effort)
    cid = state.get("cid") if isinstance(state, dict) else None
    try:
        if cid:
            root = os.path.join(os.getenv("UPLOAD_DIR", "/workspace/uploads"), "artifacts", "research", str(cid))
            latest = newest_part(root, "ledger")
            if latest:
                parts.append(
                    render_chunk(
                        f"[Ledger newest] {latest.get('path')} ~{int(latest.get('bytes') or 0)} bytes"
                    )
                )
                # Summarize older shards as one-liners
                older = list_parts(root, "ledger")[:-1]
                for p in older[-5:]:  # cap summaries
                    sha = (p.get("sha256") or "")
                    parts.append(
                        render_chunk(f"[Ledger older] {p.get('path')} sha256:{sha[:8]}")
                    )
    except Exception as exc:  # defensive logging
        log.error("icw.assembler ledger summary build failed for cid=%r: %s", cid, exc, exc_info=True)
    for c in ranked:
        parts.append(render_chunk(c))
        prompt = _joined(parts)
        if bytes_len(prompt) > in_limit_bytes:
            parts = compress_until_fit(parts, in_limit_bytes)
            prompt = _joined(parts)
            if bytes_len(prompt) > in_limit_bytes:
                parts.pop()  # defer this chunk to next window
                break
    if footer:
        parts.append(footer)
    return SimpleNamespace(prompt=_joined(parts), target_output_tokens=step_out_tokens)


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
    last_user = str(goals or "").strip()
    if not last_user:
        return None, None, {"ok": True, "reason": "empty_goals"}

    msgs = messages or []
    # Budget: token limit -> byte budget, then allocate pct_budget for ICW.
    total_budget_bytes = int(byte_budget_for_model(int(model_ctx_limit_tokens)))
    icw_budget_bytes = int(total_budget_bytes * float(pct_budget))
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
    icw_state["state_hash"] = state_hash_from(icw_state)

    req = {"role": "user", "content": last_user}
    window = assemble_window(req, icw_state, int(icw_budget_bytes), int(step_out_tokens))
    prompt = getattr(window, "prompt", "") if window is not None else ""
    prompt = str(prompt or "").strip()
    if not prompt:
        return None, None, {"ok": False, "reason": "empty_prompt"}

    pack_hash = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    frame = {"role": "system", "content": "### [ICW CONTEXT]\n" + prompt}
    meta = {
        "ok": True,
        "budget_bytes": int(icw_budget_bytes),
        "total_budget_bytes": int(total_budget_bytes),
        "pct_budget": float(pct_budget),
        "approx_chars": len(prompt),
        "state_hash": str(icw_state.get("state_hash") or ""),
    }
    return frame, pack_hash, meta

