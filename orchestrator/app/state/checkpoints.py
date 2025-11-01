from __future__ import annotations

import json
import os
from typing import Any, Dict, List


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


