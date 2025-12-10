from __future__ import annotations

import logging

log = logging.getLogger("orchestrator.icw")


def bytes_len(text: str) -> int:
    return 0 if not text else len(text.encode("utf-8"))


def byte_budget_for_model(model_ctx_limit_tokens: int, headroom_ratio: float = 0.95) -> int:
    """Compute a byte budget from a token limit with headroom.

    This is a best-effort heuristic. On any failure we fall back to the raw
    token limit but we *also* log the exception so that misconfigurations or
    bad inputs are visible in logs instead of being silently masked.
    """
    budget = int(model_ctx_limit_tokens)
    try:
        budget = int(int(model_ctx_limit_tokens) * float(headroom_ratio))
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error(
            "icw.byte_budget_for_model failed for limit=%r, headroom=%r: %s",
            model_ctx_limit_tokens,
            headroom_ratio,
            exc,
            exc_info=True,
        )
        budget = int(model_ctx_limit_tokens)
    return budget
