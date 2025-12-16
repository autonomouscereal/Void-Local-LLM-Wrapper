from __future__ import annotations

import logging
import os
import time
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

log = logging.getLogger("orchestrator.icw.windowed_solver")

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
    if not isinstance(request, dict):
        log.error("icw.windowed_solver request not dict type=%s", type(request).__name__)
        request = {}
    if not isinstance(global_state, dict):
        log.error("icw.windowed_solver global_state not dict type=%s", type(global_state).__name__)
        global_state = {}
    if model is None:
        log.error("icw.windowed_solver model is None")
        return SimpleNamespace(kind="HALT", partials=["<HALT state=\"00000000\" outcome=\"error:model_none\"/>"], state=global_state)
    try:
        ctx_tokens = int(model_ctx_limit_tokens)
    except Exception:
        ctx_tokens = 0
    if ctx_tokens <= 0:
        log.error("icw.windowed_solver invalid model_ctx_limit_tokens=%r", model_ctx_limit_tokens)
        ctx_tokens = 2048

    partials: list[str] = []
    step = 0
    in_budget = byte_budget_for_model(ctx_tokens)
    state_hashes: list[str] = []
    # Governor knobs
    step_tokens = int(step_out_tokens)
    min_tokens = max(200, int(step_out_tokens * 0.4))
    max_tokens_soft = int(step_out_tokens * 1.8)
    try:
        threshold = int(str(os.getenv("ICW_STALL_N", "3") or "3").strip() or "3")
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.windowed_solver bad ICW_STALL_N env: %s", exc, exc_info=True)
        threshold = 3
    while step < max_steps:
        step += 1
        # Keep goals/state_hash aligned to the latest user request for stable CONT/HALT tags.
        global_state["goals"] = str(request.get("content", "") or "")
        try:
            global_state["state_hash"] = state_hash_from(global_state)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("icw.windowed_solver state_hash_from failed: %s", exc, exc_info=True)
            partials.append("<HALT state=\"00000000\" outcome=\"error:state_hash\"/>")
            return SimpleNamespace(kind="HALT", partials=partials, state=global_state)
        state_hashes.append(str(global_state.get("state_hash") or ""))
        # Stall reframe
        try:
            if need_reframe(state_hashes, step, threshold=threshold):
                rf = build_reframe_prompt(
                    goal=request.get("content", ""),
                    constraints=["no caps", "json-only", "continuity-first"],
                    seen_blockers=["insufficient progress"],
                )
                global_state.setdefault("candidates", []).append(rf)
        except Exception as exc:  # pragma: no cover - defensive logging
            # ICW is best-effort; a reframe failure should not crash the solve loop.
            log.error("icw.windowed_solver reframe failed: %s", exc, exc_info=True)
        # Assemble the next window (log timings and invariants, but never log the full prompt).
        t_asm = time.perf_counter()
        try:
            window = assemble_window(request, global_state, in_budget, step_tokens)
        except Exception as exc:
            # ICW packing failure: halt deterministically with an explicit error outcome.
            log.error("icw.windowed_solver assemble_window failed: %s", exc, exc_info=True)
            partials.append(f"<HALT state=\"{global_state.get('state_hash') or '00000000'}\" outcome=\"error:assemble\"/>")
            return SimpleNamespace(kind="HALT", partials=partials, state=global_state)
        asm_ms = int((time.perf_counter() - t_asm) * 1000)
        if progress:
            try:
                progress(step, "window", {"input_bytes": bytes_len(window.prompt), "step_tokens": step_tokens})
            except Exception as exc:  # pragma: no cover - defensive logging
                # Progress is optional instrumentation; never allow it to break ICW.
                log.error("icw.windowed_solver progress(window) failed: %s", exc, exc_info=True)
        log.info(
            "icw.windowed_solver step=%s state=%s in_bytes=%s in_budget=%s step_tokens=%s asm_ms=%s cand=%s",
            step,
            str(global_state.get("state_hash") or "")[:8],
            bytes_len(window.prompt),
            int(in_budget),
            int(step_tokens),
            asm_ms,
            len(global_state.get("candidates") or []) if isinstance(global_state.get("candidates"), list) else -1,
        )
        # Provider call without wrapper timeouts; any failures surface via ProviderError.
        t_call = time.perf_counter()
        try:
            out = model_chat_with_retry(
                model,
                window.prompt,
                max_tokens=step_tokens,
            )
        except ProviderError as e:
            log.warning(
                "icw.windowed_solver provider error step=%s state=%s code=%r kind=%r msg=%r",
                step,
                str(global_state.get("state_hash") or "")[:8],
                getattr(e, "code", None),
                getattr(e, "kind", None),
                str(e),
            )
            partials.append(
                f"<HALT state=\"{global_state.get('state_hash') or '00000000'}\" outcome=\"error:{e.code or e.kind}\"/>"
            )
            return SimpleNamespace(kind="HALT", partials=partials, state=global_state)
        except Exception as exc:
            # Unexpected provider exception: never allow it to bubble.
            log.error("icw.windowed_solver provider call failed: %s", exc, exc_info=True)
            partials.append(
                f"<HALT state=\"{global_state.get('state_hash') or '00000000'}\" outcome=\"error:provider_exception\"/>"
            )
            return SimpleNamespace(kind="HALT", partials=partials, state=global_state)
        call_ms = int((time.perf_counter() - t_call) * 1000)
        text = getattr(out, "text", "")
        if not isinstance(text, str):
            text = str(text or "")
        kind, cont_hash = detect_cont_or_halt(text)
        if kind == "CONT" and cont_hash is None:
            # Synthesize CONT with tail fragment for seamless stitching
            frag = last_sentence_fragment(strip_tags(text))
            text = synthesize_cont(text, global_state["state_hash"], frag)
        partials.append(text)
        out_chars = len(strip_tags(text))
        log.info(
            "icw.windowed_solver step=%s done kind=%s out_chars=%s call_ms=%s cont_hash=%s",
            step,
            kind,
            out_chars,
            call_ms,
            (str(cont_hash)[:8] if cont_hash else ""),
        )
        # Governor: adjust step size gently
        if kind == "CONT":
            if out_chars < 200 and step_tokens < max_tokens_soft:
                step_tokens = int(step_tokens * 1.2)
            elif out_chars > (step_tokens * 6):
                step_tokens = max(min_tokens, int(step_tokens * 0.8))
        if progress:
            try:
                progress(step, "step", {"kind": kind, "out_len": out_chars, "step_tokens": step_tokens})
            except Exception as exc:  # pragma: no cover - defensive logging
                log.error("icw.windowed_solver progress(step) failed: %s", exc, exc_info=True)
        # Update state (lightweight)
        global_state["model"] = getattr(out, "model_name", "unknown")
        global_state.setdefault("window_steps", 0)
        global_state["window_steps"] += 1
        if kind == "HALT":
            return SimpleNamespace(kind="HALT", partials=partials, state=global_state)
        global_state["last_cont_hash"] = cont_hash or global_state["state_hash"]
    if progress:
        try:
            progress(step, "halt", {"reason": "max_steps"})
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("icw.windowed_solver progress(halt) failed: %s", exc, exc_info=True)
    return SimpleNamespace(kind="HALT", partials=partials, state=global_state)


