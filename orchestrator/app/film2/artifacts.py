from __future__ import annotations

import os
import json
import time
import hashlib
from typing import Dict, Any, List


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Manifest:
    def __init__(self, cid: str):
        self.cid = cid
        self.created_at = int(time.time())
        self.items: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {"cid": self.cid, "created_at": self.created_at, "items": self.items}

    def list(self) -> List[Dict[str, Any]]:
        return list(self.items)


def _root(cid: str) -> str:
    p = os.path.join(UPLOAD_DIR, "film", cid)
    os.makedirs(p, exist_ok=True)
    return p


def new_manifest(cid: str) -> Manifest:
    return Manifest(cid)


def add_artifact(manifest: Manifest, rel_path: str, payload: Any, dir: bool = False, overwrite: bool = False) -> Dict[str, Any]:
    root = _root(manifest.cid)
    full = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if dir:
        os.makedirs(full, exist_ok=True)
        item = {"path": rel_path, "kind": "dir"}
        manifest.items.append(item)
        return item
    # if payload is dict with inline content, write it; else store metadata-only
    if isinstance(payload, dict):
        # write text/json if present
        if "text" in payload and isinstance(payload["text"], str):
            tmp = full + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload["text"])
            os.replace(tmp, full)
            data = payload["text"].encode("utf-8")
            item = {"path": rel_path, "kind": "text", "hash": f"sha256:{_sha256_bytes(data)}"}
            manifest.items.append(item)
            return item
        if "timeline" in payload and isinstance(payload["timeline"], dict):
            tmp = full + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload["timeline"], ensure_ascii=False))
            os.replace(tmp, full)
            data = json.dumps(payload["timeline"], ensure_ascii=False).encode("utf-8")
            item = {"path": rel_path, "kind": "json", "hash": f"sha256:{_sha256_bytes(data)}"}
            manifest.items.append(item)
            return item
        if "path" in payload and isinstance(payload["path"], str):
            # treat as a reference-only artifact
            item = {"path": payload["path"], "kind": "file"}
            manifest.items.append(item)
            return item
    # fallback: write JSON of payload if not None
    if payload is not None:
        tmp = full + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
        os.replace(tmp, full)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        item = {"path": rel_path, "kind": "json", "hash": f"sha256:{_sha256_bytes(data)}"}
        manifest.items.append(item)
        return item
    # no-op artifact
    item = {"path": rel_path, "kind": "file"}
    manifest.items.append(item)
    return item


def save_manifest(manifest: Manifest) -> None:
    root = _root(manifest.cid)
    path = os.path.join(root, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest.to_dict(), ensure_ascii=False))
    os.replace(tmp, path)


