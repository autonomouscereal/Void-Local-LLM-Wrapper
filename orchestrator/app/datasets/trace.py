from __future__ import annotations

import os
import json
import time
from typing import Dict, Any, Optional


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")


def _public_url(path: str) -> str:
    if not isinstance(path, str) or not path:
        return ""
    if path.startswith("/workspace/"):
        rel = path.replace("/workspace", "")
    elif path.startswith("/uploads/"):
        rel = path
    else:
        # best-effort: treat as already public
        return path
    return f"{PUBLIC_BASE_URL.rstrip('/')}{rel}" if PUBLIC_BASE_URL else rel


def _trace_root() -> str:
    return os.path.join(UPLOAD_DIR, "datasets", "trace")


def append_sample(kind: str, row: Dict[str, Any]) -> str:
    """
    Append a single normalized training/trace sample for a modality.
    kind: image | tts | music | video
    Row should be compact and self-contained: inputs/refs/seeds/metrics/urls.
    Returns the path written.
    """
    root = _trace_root()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{kind}.jsonl")
    # Stamp ts and normalize any local paths to public URLs
    row = dict(row)
    row.setdefault("ts", int(time.time()))
    for key in ("path", "url", "audio_ref", "image_ref", "track_ref"):
        v = row.get(key)
        if isinstance(v, str):
            row[key] = _public_url(v)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # Maintain a tiny index of file sizes for convenience
    try:
        idx_path = os.path.join(root, "index.json")
        idx = {}
        if os.path.exists(idx_path):
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    from ..jsonio.helpers import parse_json_text as _parse_json_text
                    idx = _parse_json_text(f.read(), {})
            except Exception:
                idx = {}
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        idx[kind] = {"path": _public_url(path.replace(UPLOAD_DIR, "/uploads")), "size_bytes": int(sz)}
        tmp = idx_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(idx, ensure_ascii=False))
        os.replace(tmp, idx_path)
    except Exception:
        pass
    return path


