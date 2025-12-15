from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from app.state.checkpoints import append_ndjson


def _iso_now() -> str:
    """
    Return current UTC time in ISO 8601 format with Z suffix.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _base_dir(state_dir: str, trace_id: str) -> str:
    base = os.path.join(state_dir, "traces", str(trace_id))
    os.makedirs(base, exist_ok=True)
    return base


def _write_jsonl(state_dir: str, trace_id: str, filename: str, obj: Dict[str, Any]) -> None:
    """
    Append a single JSON object to traces/<trace_id>/<filename> as JSONL.
    Uses the existing append_ndjson helper for atomic writes.
    """
    try:
        base = _base_dir(state_dir, trace_id)
        path = os.path.join(base, filename)
        append_ndjson(path, obj)
    except Exception:
        # Trace writing is best-effort; never allow it to break request handling.
        return


def log_request(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", str(trace_id))
    rec.setdefault("timestamp", _iso_now())
    _write_jsonl(state_dir, trace_id, "requests.jsonl", rec)


def log_tool(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", str(trace_id))
    rec.setdefault("timestamp", _iso_now())
    _write_jsonl(state_dir, trace_id, "tools.jsonl", rec)


def log_event(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", str(trace_id))
    rec.setdefault("timestamp", _iso_now())
    _write_jsonl(state_dir, trace_id, "events.jsonl", rec)


def log_artifact(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", str(trace_id))
    rec.setdefault("timestamp", _iso_now())
    _write_jsonl(state_dir, trace_id, "artifacts.jsonl", rec)


def log_response(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", str(trace_id))
    rec.setdefault("timestamp", _iso_now())
    _write_jsonl(state_dir, trace_id, "responses.jsonl", rec)


def log_error(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", str(trace_id))
    rec.setdefault("timestamp", _iso_now())
    # For errors, ensure a basic error envelope exists
    if "error" not in rec:
        rec["error"] = {
            "code": str(rec.get("code") or "error"),
            "message": str(rec.get("message") or ""),
            "status": int(rec.get("status") or 0),
            "details": rec.get("details") or {},
        }
    _write_jsonl(state_dir, trace_id, "errors.jsonl", rec)


