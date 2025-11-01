from __future__ import annotations

from types import SimpleNamespace
from .assembler import assemble_window
from .continuation import detect_cont_or_halt, state_hash_from
from .tokenizer import byte_budget_for_model, bytes_len


def solve(request: dict, global_state: dict, model, model_ctx_limit_tokens: int, step_out_tokens: int = 900, max_steps: int = 64):
    """
    request: last user message dict
    global_state: dict with history, entities, candidates, planner_state, etc.
    model: object exposing .chat(prompt, max_tokens) -> SimpleNamespace(text=..., model_name=...)
    """
    partials: list[str] = []
    step = 0
    in_budget = byte_budget_for_model(model_ctx_limit_tokens)
    while step < max_steps:
        step += 1
        global_state["state_hash"] = state_hash_from(global_state)
        window = assemble_window(request, global_state, in_budget, step_out_tokens)
        assert bytes_len(window.prompt) <= in_budget
        out = model.chat(window.prompt, max_tokens=step_out_tokens)
        text = getattr(out, "text", "")
        kind, cont_hash = detect_cont_or_halt(text)
        partials.append(text)
        # Update state (lightweight; caller handles normalization and tracing)
        global_state["model"] = getattr(out, "model_name", "unknown")
        global_state.setdefault("window_steps", 0)
        global_state["window_steps"] += 1
        if kind == "HALT":
            return SimpleNamespace(kind="HALT", partials=partials, state=global_state)
        # CONT path: loop continues
        global_state["last_cont_hash"] = cont_hash or global_state["state_hash"]
    # Fallback: max steps reached
    return SimpleNamespace(kind="HALT", partials=partials, state=global_state)


