from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from app.trace_utils import emit_trace


def _iso_now() -> str:
    """
    Return current UTC time in ISO 8601 format with Z suffix.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_request(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", trace_id)
    rec.setdefault("timestamp", _iso_now())
    emit_trace(state_dir, trace_id, "request", rec)


def log_tool(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", trace_id)
    rec.setdefault("timestamp", _iso_now())
    kind = str(rec.get("event") or "tool")
    emit_trace(state_dir, trace_id, kind, rec)


def log_event(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", trace_id)
    rec.setdefault("timestamp", _iso_now())
    kind = str(rec.get("kind") or rec.get("event") or "event")
    emit_trace(state_dir, trace_id, kind, rec)


def log_artifact(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", trace_id)
    rec.setdefault("timestamp", _iso_now())
    emit_trace(state_dir, trace_id, "artifact", rec)


def log_response(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", trace_id)
    rec.setdefault("timestamp", _iso_now())
    emit_trace(state_dir, trace_id, "response", rec)


def log_error(state_dir: str, trace_id: str, record: Dict[str, Any]) -> None:
    rec = dict(record or {})
    rec.setdefault("trace_id", trace_id)
    rec.setdefault("timestamp", _iso_now())
    # For errors, ensure a basic error envelope exists
    if "error" not in rec:
        rec["error"] = {
            "code": str(rec.get("code") or "error"),
            "message": str(rec.get("message") or ""),
            "status": int(rec.get("status") or 0),
            "details": rec.get("details") or {},
        }
    emit_trace(state_dir, trace_id, "error", rec)


