from __future__ import annotations

import logging

log = logging.getLogger("orchestrator.icw")


def bytes_len(text: str) -> int:
    if not text:
        return 0
    if not isinstance(text, str):
        orig_type = type(text).__name__
        try:
            text = str(text)
            log.debug("icw.bytes_len coerced_non_str type=%s", orig_type)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("icw.bytes_len str() failed type=%s: %s", type(text), exc, exc_info=True)
            return 0
    try:
        return len(text.encode("utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.bytes_len utf-8 encode failed: %s", exc, exc_info=True)
        return 0


def byte_budget_for_model(model_ctx_limit_tokens: int, headroom_ratio: float = 0.95) -> int:
    """Compute an approximate *byte* budget from a token limit.

    IMPORTANT: ICW assembly enforces budgets using UTF-8 byte length
    (see bytes_len). The model context limit is expressed in tokens, so we
    must convert tokens → approx characters/bytes. This keeps ICW windows
    from being artificially tiny.

    Heuristic: ~4 characters per token (consistent with other parts of the codebase).
    """
    # Defensive: callers may pass str-ish limits. Never raise here.
    try:
        _limit = int(model_ctx_limit_tokens)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("icw.byte_budget_for_model bad model_ctx_limit_tokens=%r: %s", model_ctx_limit_tokens, exc, exc_info=True)
        _limit = 0
    if _limit <= 0:
        _limit = 2048
    budget = int(_limit) * 4
    try:
        budget = int(int(_limit) * 4 * float(headroom_ratio))
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error(
            "icw.byte_budget_for_model failed for limit=%r, headroom=%r: %s",
            model_ctx_limit_tokens,
            headroom_ratio,
            exc,
            exc_info=True,
        )
        budget = int(_limit) * 4
    return budget
