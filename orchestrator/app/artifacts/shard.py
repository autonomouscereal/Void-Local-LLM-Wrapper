from __future__ import annotations

import os
import json
import hashlib
import time
from typing import Optional, Dict, Any, List


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()


def _atomic_write(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _index_path(root: str, name: str) -> str:
    return os.path.join(root, f"{name}.index.json")


def _part_path(root: str, name: str, i: int) -> str:
    return os.path.join(root, f"{name}.part{i:04d}.jsonl")


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_json_atomic(path: str, obj: dict):
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def open_shard(root: str, name: str, max_bytes: int) -> dict:
    """
    Returns { 'root', 'name', 'max', 'i', 'path', 'bytes' } for appending JSONL rows.
    If current shard is full/missing, rotates to a new part (…part000X.jsonl).
    """
    os.makedirs(root, exist_ok=True)
    idx = _read_json(_index_path(root, name)) or {"name": name, "created_at": int(time.time()), "parts": []}
    # pick last part or start fresh
    i = (idx["parts"][-1]["i"] if idx["parts"] else -1) + 1
    p = _part_path(root, name, i)
    # if last exists and has room, use it; else rotate
    if os.path.exists(p):
        sz = os.path.getsize(p)
        if sz < max_bytes:
            return {"root": root, "name": name, "max": max_bytes, "i": i, "path": p, "bytes": sz}
        i += 1; p = _part_path(root, name, i)
    # new part
    _atomic_write(p, b"")
    idx["parts"].append({"i": i, "path": os.path.relpath(p, root), "bytes": 0, "sha256": None})
    _write_json_atomic(_index_path(root, name), idx)
    return {"root": root, "name": name, "max": max_bytes, "i": i, "path": p, "bytes": 0}


def append_jsonl(sh: dict, row: dict) -> dict:
    """
    Append one JSON record to the current shard; rotate if needed.
    Returns updated shard descriptor.
    """
    b = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    new_size = sh["bytes"] + len(b)
    if new_size > sh["max"]:
        # finalize current shard digest in index
        _finalize_shard(sh)
        # rotate
        i = sh["i"] + 1
        sh = {"root": sh["root"], "name": sh["name"], "max": sh["max"], "i": i, "path": _part_path(sh["root"], sh["name"], i), "bytes": 0}
        _atomic_write(sh["path"], b"")
        _touch_index(sh)
        new_size = len(b)
    with open(sh["path"], "ab") as f:
        f.write(b)
        f.flush(); os.fsync(f.fileno())
    sh["bytes"] = new_size
    _update_index_bytes(sh, new_size)
    return sh


def _touch_index(sh: dict):
    idxp = _index_path(sh["root"], sh["name"])
    idx = _read_json(idxp) or {"name": sh["name"], "created_at": int(time.time()), "parts": []}
    idx["parts"].append({"i": sh["i"], "path": os.path.relpath(sh["path"], sh["root"]), "bytes": 0, "sha256": None})
    _write_json_atomic(idxp, idx)


def _update_index_bytes(sh: dict, size: int):
    idxp = _index_path(sh["root"], sh["name"])
    idx = _read_json(idxp) or {}
    for part in idx.get("parts", []):
        if part.get("i") == sh["i"]:
            part["bytes"] = int(size)
            break
    _write_json_atomic(idxp, idx)


def _finalize_shard(sh: dict):
    # compute sha256 and freeze index entry
    try:
        with open(sh["path"], "rb") as f:
            data = f.read()
        digest = _sha256_bytes(data)
    except Exception:
        digest = None
    idxp = _index_path(sh["root"], sh["name"])
    idx = _read_json(idxp) or {}
    for part in idx.get("parts", []):
        if part.get("i") == sh.get("i"):
            part["sha256"] = digest
            part["finalized_at"] = int(time.time())
            break
    _write_json_atomic(idxp, idx)


def list_parts(root: str, name: str) -> List[dict]:
    idx = _read_json(_index_path(root, name)) or {"parts": []}
    return idx.get("parts", [])


def newest_part(root: str, name: str) -> Optional[dict]:
    parts = list_parts(root, name)
    return parts[-1] if parts else None

## duplicate block removed


