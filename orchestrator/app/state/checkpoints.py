from __future__ import annotations

import json
import os
from typing import Any, Dict, List
import time
from .ids import step_id


def _append_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "a", encoding="utf-8") as f:
        f.write(text)
    with open(tmp, "rb") as rf:
        chunk = rf.read()
    with open(path, "ab") as wf:
        wf.write(chunk)
    os.remove(tmp)


def append_ndjson(path: str, obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    _append_atomic(path, line)


def read_tail(path: str, n: int = 10) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-max(1, int(n)) :]
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    except Exception:
        return []
    return out


# ---- Step 8: append-only JSONL checkpoints ----

def _dir(root: str, key: str) -> str:
    p = os.path.join(root, key)
    os.makedirs(p, exist_ok=True)
    return p


def _path(root: str, key: str) -> str:
    return os.path.join(_dir(root, key), "trace.jsonl")


def append_event(root: str, key: str, kind: str, data: Dict[str, Any]) -> None:
    rec = {"t": int(time.time() * 1000), "step_id": step_id(), "kind": kind, "data": data or {}}
    path = _path(root, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def read_all(root: str, key: str) -> List[Dict[str, Any]]:
    try:
        with open(_path(root, key), "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def last_event(root: str, key: str, kind: str | None = None) -> Dict[str, Any] | None:
    evs = read_all(root, key)
    if kind:
        evs = [e for e in evs if e.get("kind") == kind]
    return evs[-1] if evs else None

