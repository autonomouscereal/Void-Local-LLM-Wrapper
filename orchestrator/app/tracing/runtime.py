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
    # Prefer explicit trace identifiers; fall back to conversation/job ids when present.
    raw_key = payload.get("trace_id") or payload.get("tid") or payload.get("cid") or "global"
    key = str(raw_key).strip() if isinstance(raw_key, (str, int)) else "global"
    if not key:
        key = "global"
    # Ensure trace_id is always present in event payload when we have a key.
    if key != "global" and not (isinstance(payload.get("trace_id"), str) and payload.get("trace_id")):
        payload["trace_id"] = key
    emit_trace(STATE_DIR, key, str(kind or "event"), payload)


