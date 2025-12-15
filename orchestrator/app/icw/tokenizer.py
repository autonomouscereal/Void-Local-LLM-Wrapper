from __future__ import annotations

import logging

log = logging.getLogger("orchestrator.icw")


def bytes_len(text: str) -> int:
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    return len(text.encode("utf-8"))


def byte_budget_for_model(model_ctx_limit_tokens: int, headroom_ratio: float = 0.95) -> int:
    """Compute an approximate *byte* budget from a token limit.

    IMPORTANT: ICW assembly enforces budgets using UTF-8 byte length
    (see bytes_len). The model context limit is expressed in tokens, so we
    must convert tokens → approx characters/bytes. This keeps ICW windows
    from being artificially tiny.

    Heuristic: ~4 characters per token (consistent with other parts of the codebase).
    """
    budget = int(model_ctx_limit_tokens) * 4
    try:
        budget = int(int(model_ctx_limit_tokens) * 4 * float(headroom_ratio))
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error(
            "icw.byte_budget_for_model failed for limit=%r, headroom=%r: %s",
            model_ctx_limit_tokens,
            headroom_ratio,
            exc,
            exc_info=True,
        )
        budget = int(model_ctx_limit_tokens) * 4
    return budget
