from __future__ import annotations


def bytes_len(text: str) -> int:
    return 0 if not text else len(text.encode("utf-8"))


def byte_budget_for_model(model_ctx_limit_tokens: int, headroom_ratio: float = 0.95) -> int:
    # Treat tokens as bytes; keep headroom for provider overhead.
    # This is a coarse approximation; callers should prefer smaller budgets when in doubt.
    try:
        return int(int(model_ctx_limit_tokens) * float(headroom_ratio))
    except Exception:
        return int(model_ctx_limit_tokens)


