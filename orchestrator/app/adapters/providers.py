from __future__ import annotations

from types import SimpleNamespace
from ..jsonio.normalize import normalize_to_envelope


def ask_model_json(provider, prompt: str, max_tokens: int) -> dict:
    """
    Calls provider.chat and normalizes the text reply into the canonical envelope.
    Provider must implement .chat(prompt, max_tokens) -> SimpleNamespace(text, model_name)
    """
    raw = provider.chat(prompt, max_tokens=max_tokens)
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


