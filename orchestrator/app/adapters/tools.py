from __future__ import annotations

import json
from ..jsonio.normalize import normalize_to_envelope


def run_tool_json(tool_fn, *args, **kwargs) -> dict:
    """
    Run a tool, normalize its textual JSON output to the canonical envelope.
    Tools should emit JSON; if they return plain text or objects, normalize anyway.
    """
    raw = tool_fn(*args, **kwargs)
    if isinstance(raw, str):
        text = raw
    else:
        try:
            text = json.dumps(raw, ensure_ascii=False)
        except Exception:
            text = str(raw)
    env = normalize_to_envelope(text)
    return env


