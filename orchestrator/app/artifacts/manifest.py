from __future__ import annotations

import os
import json
import time
import hashlib
from typing import Dict, Any


def _sha(path: str):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def add_manifest_row(manifest: Dict[str, Any], path: str, step_id: str):
    created_ms = int(time.time() * 1000)
    row = {
        "path": path,
        "bytes": (os.path.getsize(path) if os.path.exists(path) else 0),
        "sha256": _sha(path),
        "step_id": step_id,
        "created_at": int(created_ms // 1000),
        "created_ms": int(created_ms),
    }
    manifest.setdefault("items", []).append(row)
    return row


def write_manifest_atomic(dirpath: str, manifest: Dict[str, Any]):
    os.makedirs(dirpath, exist_ok=True)
    tmp = os.path.join(dirpath, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, os.path.join(dirpath, "manifest.json"))


