from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
import time
import socket
from ..jsonio.normalize import normalize_to_envelope


class ProviderError(Exception):
    def __init__(self, kind: str, code: str | None = None, msg: str = ""):
        super().__init__(msg)
        self.kind = kind
        self.code = code
        self.msg = msg


RETRYABLE_CODES = {"429", "408", "502", "503", "504"}


def _str(e: Exception) -> str:
    try:
        return str(e)
    except Exception:
        return ""


def classify_exc(e: Exception) -> ProviderError:
    if isinstance(e, TimeoutError) or isinstance(e, socket.timeout):
        return ProviderError("retryable", "408", "timeout")
    s = _str(e)
    if "429" in s or "rate" in s.lower():
        return ProviderError("retryable", "429", "rate limited")
    if any(x in s for x in ("502", "503", "504", "overload")):
        return ProviderError("retryable", "503", "provider overloaded")
    return ProviderError("permanent", None, s)


def _call_provider_chat(provider, prompt: str, max_tokens: int, timeout: int | None = None):
    try:
        if timeout is not None:
            return provider.chat(prompt, max_tokens=max_tokens, timeout=timeout)
        return provider.chat(prompt, max_tokens=max_tokens)
    except TypeError:
        # Providers without timeout param
        return provider.chat(prompt, max_tokens=max_tokens)


def model_chat_with_retry(provider, prompt: str, max_tokens: int, timeout_s: int = 30, retries: int = 2, backoff: List[int] | None = None):
    if backoff is None:
        backoff = [0, 2, 4]
    last_err: ProviderError | None = None
    for attempt in range(retries + 1):
        try:
            resp = _call_provider_chat(provider, prompt, max_tokens, timeout=timeout_s)
            return resp
        except Exception as e:
            perr = classify_exc(e)
            last_err = perr
            if (perr.kind != "retryable") or (attempt == retries):
                raise perr
            # deterministic backoff
            delay = backoff[min(attempt + 1, len(backoff) - 1)] if backoff else 0
            if delay > 0:
                time.sleep(delay)
            continue
    raise last_err if last_err else ProviderError("permanent", None, "unknown")


def ask_model_json(provider, prompt: str, max_tokens: int) -> dict:
    """
    Calls provider.chat and normalizes the text reply into the canonical envelope.
    Provider must implement .chat(prompt, max_tokens) -> SimpleNamespace(text, model_name)
    """
    raw = _call_provider_chat(provider, prompt, max_tokens)
    if isinstance(raw, SimpleNamespace):
        text = getattr(raw, "text", "")
        model_name = getattr(raw, "model_name", "unknown")
    else:
        text = getattr(raw, "text", "")
        model_name = getattr(raw, "model_name", "unknown")
    env = normalize_to_envelope(text)
    env.setdefault("meta", {})
    env["meta"]["model"] = model_name
    return env


