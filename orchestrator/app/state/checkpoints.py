from __future__ import annotations

import json
import os
from typing import Any, Dict, List
import time
import logging
from datetime import datetime, timezone
from .ids import step_id
from ..json_parser import JSONParser


log = logging.getLogger(__name__)
def _append_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "a", encoding="utf-8") as f:
            f.write(text)
        with open(tmp, "rb") as rf:
            chunk = rf.read()
        with open(path, "ab") as wf:
            wf.write(chunk)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            # Best-effort cleanup; never raise from trace persistence.
            log.debug("checkpoints._append_atomic: failed to cleanup tmp=%s", tmp, exc_info=True)


def append_ndjson(path: str, obj: Dict[str, Any]) -> None:
    try:
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
        _append_atomic(path, line)
    except Exception as ex:
        # Trace/checkpoint I/O must never break request handling.
        log.warning("checkpoints.append_ndjson failed for path=%s: %s", path, ex, exc_info=True)
        return


def read_tail(path: str, n: int = 10) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    parser = JSONParser()
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-max(1, int(n)) :]
    schema = {"t": int, "step_id": str, "kind": str, "data": dict}
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        obj = parser.parse(ln, schema)
        if isinstance(obj, dict):
            out.append(obj)
    return out


# ---- Step 8: append-only JSONL checkpoints ----

def _iso_now() -> str:
    # UTC ISO8601 with Z suffix (stable, human-readable).
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_root(root: str) -> str:
    """
    Unify trace root layout across the codebase.

    Historically some callers used:
      - <state>/traces/<trace_id>/trace.jsonl
    while others used:
      - <state>/<trace_id>/trace.jsonl

    We standardize on <state>/traces/<trace_id>/trace.jsonl.
    """
    r = (root or "").rstrip("/\\")
    if not r:
        return root
    base = os.path.basename(r).lower()
    if base == "traces":
        return r
    return os.path.join(r, "traces")


def _infer_domain(kind: str) -> str:
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


def _dir(root: str, key: str) -> str:
    root2 = _normalize_root(root)
    p = os.path.join(root2, key)
    os.makedirs(p, exist_ok=True)
    return p


def _path(root: str, key: str) -> str:
    return os.path.join(_dir(root, key), "trace.jsonl")


def append_event(root: str, key: str, kind: str, data: Dict[str, Any]) -> None:
    try:
        # Normalize and enrich the payload so all trace writers converge on the
        # same schema/layout regardless of call site.
        d = dict(data or {})
        d.setdefault("trace_id", str(key))
        d.setdefault("ts_iso", _iso_now())
        d.setdefault("domain", _infer_domain(kind))
        rec = {"t": int(time.time() * 1000), "step_id": step_id(), "kind": kind, "data": d}
        path = _path(root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception as ex:
                # Fsync failure shouldn't break request handling, but we still want
                # to see it in logs instead of swallowing it.
                log.warning("checkpoints.append_event: fsync failed for path=%s: %s", path, ex, exc_info=True)
    except Exception as ex:
        log.warning("checkpoints.append_event failed for key=%s kind=%s: %s", key, kind, ex, exc_info=True)
        return


def read_all(root: str, key: str) -> List[Dict[str, Any]]:
    path = _path(root, key)
    if not os.path.exists(path):
        # Back-compat: older traces were written under <state>/<trace_id>/trace.jsonl
        # (without the intermediate "traces" directory).
        legacy_root = (root or "").rstrip("/\\")
        if legacy_root and os.path.basename(legacy_root).lower() != "traces":
            legacy_path = os.path.join(legacy_root, key, "trace.jsonl")
            if os.path.exists(legacy_path):
                path = legacy_path
            else:
                return []
        else:
            return []
    parser = JSONParser()
    out: List[Dict[str, Any]] = []
    schema = {"t": int, "step_id": str, "kind": str, "data": dict}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            obj = parser.parse(ln, schema)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def last_event(root: str, key: str, kind: str | None = None) -> Dict[str, Any] | None:
    evs = read_all(root, key)
    if kind:
        evs = [e for e in evs if e.get("kind") == kind]
    return evs[-1] if evs else None

