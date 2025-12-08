from __future__ import annotations

from typing import Any, Dict


def build_openai_envelope(
    *,
    ok: bool,
    text: str,
    error: Dict[str, Any] | None,
    usage: Dict[str, Any],
    model: str,
    seed: int,
    id_: str,
) -> Dict[str, Any]:
    """
    Canonical OpenAI-style chat.completion envelope builder.

    This is used by orchestrator endpoints that expose OpenAI-compatible
    responses on top of the internal ToolEnvelope / committee pipeline.
    """
    return {
        "id": id_ or "orc-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": (text or "(no content)")},
            }
        ],
        "usage": usage,
        "ok": bool(ok),
        "error": (error or None),
        "seed": seed,
    }


