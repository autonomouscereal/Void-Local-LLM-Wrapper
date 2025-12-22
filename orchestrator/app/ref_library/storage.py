from __future__ import annotations

import os
import json
import time
import hashlib
from ..json_parser import JSONParser


# Reference-library storage root (separate concept from lock bundles).
REF_ROOT = os.getenv("REF_ROOT", os.path.join("/workspace", "uploads", "artifacts", "refs"))


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(65536), b""):
            h.update(ch)
    return h.hexdigest()


def ref_dir(ref_id: str) -> str:
    return os.path.join(REF_ROOT, ref_id)


def write_atomic(path: str, data: bytes | str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def new_ref_id(kind: str) -> str:
    t = int(time.time() * 1000)
    rid = f"{kind}-{t:x}"
    os.makedirs(ref_dir(rid), exist_ok=True)
    return rid


def save_manifest(ref_id: str, manifest: dict) -> None:
    write_atomic(os.path.join(ref_dir(ref_id), "manifest.json"), json.dumps(manifest, ensure_ascii=False, indent=2))


def load_manifest(ref_id: str) -> dict | None:
    p = os.path.join(ref_dir(ref_id), "manifest.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            parser = JSONParser()
            # Manifests are stored as arbitrary dicts; coerce to generic mapping.
            data = parser.parse(f.read(), {})
            return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None


