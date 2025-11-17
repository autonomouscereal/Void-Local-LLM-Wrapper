from __future__ import annotations

import os
import json
import glob
from typing import List, Dict, Any
from .registry import _sha


def _g(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern, recursive=True))


def _copy(src: str, dst: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    import shutil
    shutil.copy2(src, dst)
    return {"path": dst, "bytes": os.path.getsize(dst), "sha256": _sha(dst)}


def _concat_jsonl(files: List[str], out: str) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as w:
        for p in files:
            with open(p, "rb") as r:
                for line in r:
                    w.write(line)


def export_facts(dst_dir: str) -> Dict[str, Any]:
    files = _g("**/facts_*.jsonl") + _g("**/facts.jsonl")
    files = sorted(list(dict.fromkeys(files)))
    if not files:
        return {}
    out = os.path.join(dst_dir, "facts.jsonl")
    _concat_jsonl(files, out)
    return {"file": out, "bytes": os.path.getsize(out), "sha256": _sha(out), "sources": [{"path": p, "bytes": os.path.getsize(p), "sha256": _sha(p)} for p in files]}


def export_image_samples(dst_dir: str) -> Dict[str, Any]:
    files = _g("**/image_samples_*.jsonl") + _g("**/image_samples.jsonl")
    files = sorted(list(dict.fromkeys(files)))
    if not files:
        return {}
    out = os.path.join(dst_dir, "image_samples.jsonl")
    _concat_jsonl(files, out)
    return {"file": out, "bytes": os.path.getsize(out), "sha256": _sha(out), "sources": [{"path": p, "bytes": os.path.getsize(p), "sha256": _sha(p)} for p in files]}


def export_tts_samples(dst_dir: str) -> Dict[str, Any]:
    files = _g("**/tts_samples_*.jsonl") + _g("**/tts_samples.jsonl")
    files = sorted(list(dict.fromkeys(files)))
    if not files:
        return {}
    out = os.path.join(dst_dir, "tts_samples.jsonl")
    _concat_jsonl(files, out)
    return {"file": out, "bytes": os.path.getsize(out), "sha256": _sha(out), "sources": [{"path": p, "bytes": os.path.getsize(p), "sha256": _sha(p)} for p in files]}


def export_music_samples(dst_dir: str) -> Dict[str, Any]:
    files = _g("**/music_samples_*.jsonl") + _g("**/music_samples.jsonl")
    files = sorted(list(dict.fromkeys(files)))
    if not files:
        return {}
    out = os.path.join(dst_dir, "music_samples.jsonl")
    _concat_jsonl(files, out)
    return {"file": out, "bytes": os.path.getsize(out), "sha256": _sha(out), "sources": [{"path": p, "bytes": os.path.getsize(p), "sha256": _sha(p)} for p in files]}


def export_code_patches(dst_dir: str) -> Dict[str, Any]:
    files = _g("**/code_patches_*.jsonl")
    files = sorted(list(dict.fromkeys(files)))
    if not files:
        return {}
    out = os.path.join(dst_dir, "code_patches.jsonl")
    _concat_jsonl(files, out)
    return {"file": out, "bytes": os.path.getsize(out), "sha256": _sha(out), "sources": [{"path": p, "bytes": os.path.getsize(p), "sha256": _sha(p)} for p in files]}


def export_research(dst_dir: str) -> Dict[str, Any]:
    families = []
    for idx in _g(os.path.join("/workspace", "uploads", "artifacts", "research", "**", "ledger.index.json")):
        base = os.path.dirname(idx)
        parts: Dict[str, Any] = {}
        from ..json_parser import JSONParser
        with open(idx, "r", encoding="utf-8") as fh:
            parser = JSONParser()
            parsed = parser.parse(fh.read(), {"parts": list})
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


