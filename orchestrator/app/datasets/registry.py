from __future__ import annotations

import os
import json
import time
import hashlib


ROOT = os.getenv("DATASETS_ROOT", os.path.join("/workspace", "uploads", "artifacts", "datasets"))


def _ds_dir(name: str) -> str:
    return os.path.join(ROOT, name)


def _ver_dir(name: str, version: str) -> str:
    return os.path.join(_ds_dir(name), version)


def _idx_path(name: str, version: str) -> str:
    return os.path.join(_ver_dir(name, version), "index.json")


def _sha(path: str):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for ch in iter(lambda: f.read(65536), b""):
                h.update(ch)
        return h.hexdigest()
    except Exception:
        return None


def write_index(name: str, version: str, index: dict):
    os.makedirs(_ver_dir(name, version), exist_ok=True)
    tmp = _idx_path(name, version) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(index, ensure_ascii=False, indent=2))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, _idx_path(name, version))


def list_datasets() -> list:
    if not os.path.exists(ROOT):
        return []
    return sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])


def list_versions(name: str) -> list:
    d = _ds_dir(name)
    if not os.path.exists(d):
        return []
    return sorted([v for v in os.listdir(d) if os.path.isdir(os.path.join(d, v))], reverse=True)


def new_version() -> str:
    return time.strftime("%Y-%m-%d_%H%M", time.gmtime())


