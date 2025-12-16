from __future__ import annotations

import os
from typing import Any, Dict

from ..trace_utils import emit_trace


STATE_DIR = os.path.join(os.getenv("UPLOAD_DIR", "/workspace/uploads"), "state")


def trace_event(kind: str, payload: Dict[str, Any] | Any) -> None:
    """
    Append a runtime/logging trace event into the unified per-trace stream:
      <state>/traces/<trace_id>/trace.jsonl

    This is the only API that should be used for runtime tracing across modules.
    """
    if not isinstance(payload, dict):
        payload = {"_raw": payload}
    trace_id = payload.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        # Strict mode: no fallbacks and no derived trace ids.
        return
    emit_trace(STATE_DIR, trace_id, str(kind or "event"), payload)


