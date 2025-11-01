from __future__ import annotations

from types import SimpleNamespace
from .tokenizer import bytes_len
from .compress import (
    cleanup_lossless,
    compact_numbers_tables,
    summarize_block_extractive,
    facts_pocket,
    attach_pocket,
)
from .relevance import score_chunk
from .continuation import make_system_hint


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


def assemble_window(request_msg: dict, state: dict, in_limit_bytes: int, step_out_tokens: int):
    """Return a prompt string that fits the byte budget."""
    goal = request_msg.get("content", "")
    anchor = state.get("anchor_text", "")  # recent exact turns (verbatim)
    header = make_entity_header({"entities": state.get("entities", []), "goals": goal})
    # Candidates = history chunks + RAG chunks + artifacts summaries (strings)
    candidates = list(state.get("candidates", []) or [])
    # Rank by relevance
    ranked = sorted(candidates, key=lambda t: score_chunk(t, goal), reverse=True)
    # Build parts in priority order
    system_hint = make_system_hint(step_out_tokens, state.get("state_hash", ""))
    parts: list[str] = [system_hint, render_anchor(anchor), render_header(header)]
    for c in ranked:
        parts.append(render_chunk(c))
        prompt = _joined(parts)
        if bytes_len(prompt) > in_limit_bytes:
            parts = compress_until_fit(parts, in_limit_bytes)
            prompt = _joined(parts)
            if bytes_len(prompt) > in_limit_bytes:
                parts.pop()  # defer this chunk to next window
                break
    return SimpleNamespace(prompt=_joined(parts), target_output_tokens=step_out_tokens)


