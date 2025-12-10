from __future__ import annotations

import os
from types import SimpleNamespace

from .assembler import assemble_window
from .continuation import (
    detect_cont_or_halt,
    state_hash_from,
    last_sentence_fragment,
    synthesize_cont,
    strip_tags,
)
from .tokenizer import byte_budget_for_model, bytes_len
from .reframe import need_reframe, build_reframe_prompt
from ..adapters.providers import model_chat_with_retry, ProviderError


def solve(
    request: dict,
    global_state: dict,
    model,
    model_ctx_limit_tokens: int,
    step_out_tokens: int = 900,
    max_steps: int = 64,
    progress=None,
):
    """
    request: last user message dict
    global_state: dict with history, entities, candidates, planner_state, etc.
    model: object exposing .chat(prompt, max_tokens) -> SimpleNamespace(text=..., model_name=...)
    progress: optional callback(step, kind, info)
    """
    partials: list[str] = []
    step = 0
    in_budget = byte_budget_for_model(model_ctx_limit_tokens)
    state_hashes: list[str] = []
    # Governor knobs
    step_tokens = int(step_out_tokens)
    min_tokens = max(200, int(step_out_tokens * 0.4))
    max_tokens_soft = int(step_out_tokens * 1.8)
    threshold = int(os.getenv("ICW_STALL_N", "3") or 3)
    while step < max_steps:
        step += 1
        # Keep goals/state_hash aligned to the latest user request for stable CONT/HALT tags.
        global_state["goals"] = request.get("content", "")
        global_state["state_hash"] = state_hash_from(global_state)
        state_hashes.append(global_state["state_hash"])
        # Stall reframe
        if need_reframe(state_hashes, step, threshold=threshold):
            rf = build_reframe_prompt(
                goal=request.get("content", ""),
                constraints=["no caps", "json-only", "continuity-first"],
                seen_blockers=["insufficient progress"],
            )
            global_state.setdefault("candidates", []).append(rf)
        window = assemble_window(request, global_state, in_budget, step_tokens)
        if progress:
            progress(step, "window", {"input_bytes": bytes_len(window.prompt), "step_tokens": step_tokens})
        # Provider call without wrapper timeouts; any failures surface via ProviderError.
        try:
            out = model_chat_with_retry(
                model,
                window.prompt,
                max_tokens=step_tokens,
            )
        except ProviderError as e:
            partials.append(f"<HALT state=\"{global_state['state_hash']}\" outcome=\"error:{e.code or e.kind}\"/>")
            return SimpleNamespace(kind="HALT", partials=partials, state=global_state)
        text = getattr(out, "text", "")
        kind, cont_hash = detect_cont_or_halt(text)
        if kind == "CONT" and cont_hash is None:
            # Synthesize CONT with tail fragment for seamless stitching
            frag = last_sentence_fragment(strip_tags(text))
            text = synthesize_cont(text, global_state["state_hash"], frag)
        partials.append(text)
        # Governor: adjust step size gently
        out_chars = len(strip_tags(text))
        if kind == "CONT":
            if out_chars < 200 and step_tokens < max_tokens_soft:
                step_tokens = int(step_tokens * 1.2)
            elif out_chars > (step_tokens * 6):
                step_tokens = max(min_tokens, int(step_tokens * 0.8))
        if progress:
            progress(step, "step", {"kind": kind, "out_len": out_chars, "step_tokens": step_tokens})
        # Update state (lightweight)
        global_state["model"] = getattr(out, "model_name", "unknown")
        global_state.setdefault("window_steps", 0)
        global_state["window_steps"] += 1
        if kind == "HALT":
            return SimpleNamespace(kind="HALT", partials=partials, state=global_state)
        global_state["last_cont_hash"] = cont_hash or global_state["state_hash"]
    if progress:
        progress(step, "halt", {"reason": "max_steps"})
    return SimpleNamespace(kind="HALT", partials=partials, state=global_state)


