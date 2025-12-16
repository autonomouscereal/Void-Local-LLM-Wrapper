from __future__ import annotations

import os
from typing import Any, Dict
import logging

from .state.checkpoints import append_event as checkpoints_append_event

log = logging.getLogger(__name__)


class TraceEmitter:
    def __init__(self, state_dir: str):
        self.state_dir = state_dir

    @staticmethod
    def infer_domain(kind: str, payload: Dict[str, Any] | None = None) -> str:
        k = (kind or "").lower()
        if k in ("request", "response"):
            return "api"
        if k.startswith("chat."):
            return "chat"
        if k.startswith("artifact"):
            return "artifact"
        if k.startswith("error"):
            return "error"
        if k.startswith("committee."):
            return "committee"
        if k.startswith("planner.") or k.startswith("replan."):
            return "planner"
        if k.startswith("comfy.") or "[comfy]" in k:
            return "comfy"
        if k.startswith("exec_") or k.startswith("exec.") or k.startswith("tools.exec") or k.startswith("tool.run"):
            return "executor"
        if k.startswith("tool"):
            return "tool"
        if k.startswith("memory"):
            return "memory"
        return "orchestrator"

    def emit(self, trace_id: str, kind: str, payload: Dict[str, Any]) -> None:
        data = dict(payload or {})
        data.setdefault("domain", self.infer_domain(kind, data))
        checkpoints_append_event(self.state_dir, str(trace_id), kind, data)


def infer_domain(kind: str, payload: Dict[str, Any] | None = None) -> str:
    return TraceEmitter.infer_domain(kind, payload)


def emit_trace(state_dir: str, trace_id: str, kind: str, payload: Dict[str, Any]) -> None:
    TraceEmitter(state_dir).emit(trace_id, kind, payload or {})


def append_jsonl_compat(state_dir: str, path: str, obj: Dict[str, Any]) -> None:
    """
    Back-compat JSONL writer that normalizes legacy per-trace files into the unified trace stream.
    Non-trace paths fall back to direct file append.
    """
    norm = (path or "").replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    if "traces" in parts:
        try:
            i = parts.index("traces")
            key = parts[i + 1] if len(parts) > i + 1 else "global"
            fname = parts[-1].lower()
            if fname == "responses.jsonl":
                emit_trace(state_dir, key, "response", obj or {})
                return
            if fname == "events.jsonl":
                kind = str((obj.get("kind") or obj.get("event") or "event"))
                payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
                emit_trace(state_dir, key, kind, payload or {})
                return
            if fname == "artifacts.jsonl":
                emit_trace(state_dir, key, "artifact", obj or {})
                return
            if fname == "tools.jsonl":
                ev = str((obj.get("event") or "tool"))
                emit_trace(state_dir, key, ev, obj or {})
                return
            if fname == "errors.jsonl":
                emit_trace(state_dir, key, "error", obj or {})
                return
            if fname == "chat.jsonl":
                emit_trace(state_dir, key, "chat.append", obj or {})
                return
            if fname == "requests.jsonl":
                emit_trace(state_dir, key, "request", obj or {})
                return
            if fname == "trace.jsonl":
                kind = str((obj.get("kind") or obj.get("event") or "event"))
                payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
                emit_trace(state_dir, key, kind, payload or {})
                return
        except Exception:
            # If normalization fails, fall back to direct append below.
            log.debug("append_jsonl_compat: normalization failed for path=%s", path, exc_info=True)
    # Fallback: direct append without normalization
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        import json as _json
        f.write(_json.dumps(obj, ensure_ascii=False) + "\n")


