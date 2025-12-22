from __future__ import annotations

import os
import json
import glob
import shutil
from typing import List, Dict, Any
from .registry import _sha
from .stream import STREAM_ROOT, STREAM_NAME
from ..json_parser import JSONParser


def _g(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern, recursive=True))


def _copy(src: str, dst: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": dst, "bytes": os.path.getsize(dst), "sha256": _sha(dst)}


def export_research(dst_dir: str) -> Dict[str, Any]:
    families = []
    for idx in _g(os.path.join("/workspace", "uploads", "artifacts", "research", "**", "ledger.index.json")):
        base = os.path.dirname(idx)
        parts: Dict[str, Any] = {}
        with open(idx, "r", encoding="utf-8") as fh:
            parser = JSONParser()
            schema = {"parts": list}
            parsed = parser.parse(fh.read(), schema)
            parts = parsed if isinstance(parsed, dict) else {}
        fam = {"root": base, "index": _copy(idx, os.path.join(dst_dir, os.path.relpath(idx, "/workspace/uploads/artifacts/research"))), "parts": []}
        for pr in (parts.get("parts") or []):
            src = os.path.join(base, pr.get("path"))
            dst = os.path.join(dst_dir, os.path.relpath(src, "/workspace/uploads/artifacts/research"))
            fam["parts"].append(_copy(src, dst))
        for leaf in ("money_map.json", "timeline.json", "report.json"):
            p = os.path.join(base, leaf)
            if os.path.exists(p):
                _copy(p, os.path.join(dst_dir, os.path.relpath(p, "/workspace/uploads/artifacts/research")))
        families.append(fam)
    return {"families": families}


def export_dataset_stream(dst_dir: str) -> Dict[str, Any]:
    """
    Export the canonical dataset stream by copying its index + shard parts into the dataset version.
    This avoids expensive globbing and keeps a single source of truth.
    """
    src_root = STREAM_ROOT
    idx = os.path.join(src_root, f"{STREAM_NAME}.index.json")
    if not os.path.exists(idx):
        return {}
    out_root = os.path.join(dst_dir, "dataset_stream")
    items: List[Dict[str, Any]] = []
    # copy index
    items.append(_copy(idx, os.path.join(out_root, os.path.basename(idx))))
    # copy meta (if present)
    meta_path = os.path.join(src_root, f"{STREAM_NAME}.meta.json")
    if os.path.exists(meta_path):
        items.append(_copy(meta_path, os.path.join(out_root, os.path.basename(meta_path))))
    # copy all parts referenced by index
    try:
        with open(idx, "r", encoding="utf-8") as f:
            parser = JSONParser()
            parsed = parser.parse(f.read(), {"parts": list})
        parts = (parsed.get("parts") or []) if isinstance(parsed, dict) else []
    except Exception:
        parts = []
    for pr in parts:
        if not isinstance(pr, dict):
            continue
        rel = pr.get("path")
        if not isinstance(rel, str):
            continue
        src = os.path.join(src_root, rel)
        if not os.path.exists(src):
            continue
        items.append(_copy(src, os.path.join(out_root, os.path.basename(src))))
    return {"root": out_root, "items": items}


