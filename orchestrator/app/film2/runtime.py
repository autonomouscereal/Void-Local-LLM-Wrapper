from __future__ import annotations

"""
Film2 runtime helpers.

These helpers are Film2-specific and intentionally live under `app/film2/` (not `main.py`).
They are used by the in-process Film2 tool path (`film2.run`) to:
- emit consistent Film2 trace events into the orchestrator run ledger
- log compact Film2 progress lines (high observability)
- register Film2 artifacts (e.g. rebuilt clip videos) into the run ledger
"""

import logging
import os
import time
from typing import Any, Callable, Dict, Optional


log = logging.getLogger(__name__)


def _compact_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for k in ("shot_id", "scene_id", "segment_id", "adapter", "prompt", "src", "image", "refine_mode"):
        v = payload.get(k)
        if v is None:
            continue
        if k == "prompt" and isinstance(v, str) and len(v) > 160:
            compact[k] = v[:160] + "..."
        else:
            compact[k] = v
    return compact


def film2_trace_event(
    trace_id: Optional[str],
    e: Dict[str, Any],
    *,
    log_fn: Callable[..., None],
) -> None:
    """
    Emit a Film2 trace event into the run ledger via `log_fn` (usually main._log),
    and also mirror a compact line to standard logs.
    """
    if not trace_id:
        return
    row = {"t": int(time.time() * 1000), **(e or {})}
    ev = str(row.get("event") or "event")
    payload = {k: v for k, v in row.items() if k != "event"}
    log.info("film2.event trace_id=%s event=%s fields=%s", trace_id, ev, _compact_fields(payload))
    log_fn(ev, trace_id=trace_id, **payload)


def film2_artifact_video(
    trace_id: Optional[str],
    path: str,
    *,
    upload_dir: str,
    log_fn: Callable[..., None],
) -> None:
    """
    Register a video artifact (local filesystem path) against the current trace.
    """
    if not (trace_id and isinstance(path, str) and path):
        return
    rel = os.path.relpath(path, upload_dir).replace("\\", "/")
    log_fn("artifact", trace_id=trace_id, kind="video", path=rel)




