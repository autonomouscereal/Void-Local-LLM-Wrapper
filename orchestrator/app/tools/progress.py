from __future__ import annotations

from typing import Optional, Dict, Any
import asyncio

_progress_q: Optional[asyncio.Queue] = None


def set_progress_queue(q: Optional[asyncio.Queue]) -> None:
    global _progress_q
    _progress_q = q


def get_progress_queue() -> Optional[asyncio.Queue]:
    return _progress_q


def emit_progress(event: Dict[str, Any]) -> None:
    q = _progress_q
    if q is not None and isinstance(event, dict):
        try:
            q.put_nowait(event)
        except Exception:
            pass


