from __future__ import annotations

"""
Canonical dataset stream (single source of truth).

Design goals:
- One logical dataset stream for *everything* we want to distill/train on.
- Scales safely: sharded JSONL + index, not one gigantic file.
- Extremely detailed rows: each row is self-describing, tagged, and can vary by modality.

This replaces:
- per-tool `*_samples.jsonl` scattered under artifact dirs
- `datasets/trace/*.jsonl` modality split files
- glob-based dataset exporters that sweep the workspace
"""

import os
import time
import uuid
import json
import logging
from typing import Any, Dict, Optional, List, Tuple

from ..artifacts.shard import open_shard, append_jsonl
from ..json_parser import JSONParser


log = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")

# Stream storage (under uploads/datasets/stream/)
STREAM_ROOT = os.getenv("DATASETS_STREAM_ROOT", os.path.join(UPLOAD_DIR, "datasets", "stream"))
STREAM_NAME = os.getenv("DATASETS_STREAM_NAME", "dataset")
STREAM_SHARD_BYTES = int(os.getenv("DATASETS_STREAM_SHARD_BYTES", os.getenv("ARTIFACT_SHARD_BYTES", "50000000")))

_SHARD: Optional[dict] = None


def _atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _public_url(path: str) -> str:
    """
    Convert a local filesystem path under UPLOAD_DIR into a public URL (/uploads/...).
    """
    if not isinstance(path, str) or not path:
        return ""
    if path.startswith("/workspace/"):
        rel = path.replace("/workspace", "")
    elif path.startswith("/uploads/"):
        rel = path
    else:
        # If it's already a URL-ish or non-standard path, return as-is.
        return path
    return f"{PUBLIC_BASE_URL.rstrip('/')}{rel}" if PUBLIC_BASE_URL else rel


def _norm_tags(tags: Any) -> List[str]:
    """
    Normalize tags to a stable list[str] with namespace-friendly strings.
    """
    out: List[str] = []
    if tags is None:
        return out
    if isinstance(tags, str):
        tags = [tags]
    if isinstance(tags, dict):
        # Convert mapping into ns:key=value strings
        for k, v in tags.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, (str, int, float, bool)):
                out.append(f"{k}={v}")
            elif v is None:
                out.append(k)
        return sorted(list(dict.fromkeys([t.strip() for t in out if isinstance(t, str) and t.strip()])))
    if isinstance(tags, list):
        for t in tags:
            if isinstance(t, (str, int, float, bool)):
                s = str(t).strip()
                if s:
                    out.append(s)
    return sorted(list(dict.fromkeys(out)))


def _extract_outputs(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort normalize output paths/urls into a consistent structure.
    """
    outputs: Dict[str, Any] = {}
    files: List[Dict[str, Any]] = []
    for key in ("path", "url", "audio_ref", "image_ref", "track_ref", "video_ref", "video_path", "image_path", "audio_path"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            outputs[key] = v
            if key.endswith("_ref") or key.endswith("_path") or key == "path":
                files.append({"role": key, "path": v, "url": _public_url(v)})
            if key == "url":
                files.append({"role": "url", "url": v})
    if files:
        outputs["files"] = files
    return outputs


def _meta_path() -> str:
    return os.path.join(STREAM_ROOT, f"{STREAM_NAME}.meta.json")


def _update_meta(kind: str, payload: Dict[str, Any]) -> None:
    """
    Maintain a compact meta index for discoverability:
    - counts by kind/modality/tool
    - tag counts (top-level only)
    - first/last timestamps
    """
    mp = _meta_path()
    meta: Dict[str, Any] = {}
    try:
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as f:
                parser = JSONParser()
                meta = parser.parse(f.read(), {}) if f.readable() else {}
            if not isinstance(meta, dict):
                meta = {}
    except Exception:
        meta = {}

    now_ms = int(payload.get("t") or int(time.time() * 1000))
    meta.setdefault("name", STREAM_NAME)
    meta.setdefault("root", STREAM_ROOT)
    meta.setdefault("created_at_ms", now_ms)
    meta["updated_at_ms"] = now_ms
    meta.setdefault("counts", {})
    counts = meta.get("counts") if isinstance(meta.get("counts"), dict) else {}
    meta["counts"] = counts
    counts["total"] = int(counts.get("total") or 0) + 1
    counts[f"kind:{kind}"] = int(counts.get(f"kind:{kind}") or 0) + 1
    mod = payload.get("modality")
    if isinstance(mod, str) and mod:
        counts[f"modality:{mod}"] = int(counts.get(f"modality:{mod}") or 0) + 1
    tool = payload.get("tool")
    if isinstance(tool, str) and tool:
        counts[f"tool:{tool}"] = int(counts.get(f"tool:{tool}") or 0) + 1

    # Tag counts (capped)
    tag_counts = meta.get("tag_counts") if isinstance(meta.get("tag_counts"), dict) else {}
    meta["tag_counts"] = tag_counts
    tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
    for t in tags[:64]:
        if not isinstance(t, str):
            continue
        k = t[:120]
        tag_counts[k] = int(tag_counts.get(k) or 0) + 1

    # Document a schema version for this stream
    meta.setdefault("schema_version", 1)
    meta.setdefault(
        "row_schema",
        {
            "t": "int ms",
            "id": "string uuid",
            "kind": "string",
            "modality": "string? (for training_sample)",
            "conversation_id": "string?",
            "trace_id": "string?",
            "tool": "string?",
            "tags": "list[string]",
            "inputs": "dict?",
            "outputs": "dict? (includes outputs.files[] with url/path)",
            "locks": "dict?",
            "qa": "dict?",
            "metrics": "dict?",
            "meta": "dict?",
            "payload": "dict (original row)",
        },
    )
    try:
        _atomic_write_text(mp, json.dumps(meta, ensure_ascii=False, indent=2))
    except Exception:
        # meta is helpful but never critical
        return


def _ensure_shard() -> Optional[dict]:
    global _SHARD
    if _SHARD is not None:
        return _SHARD
    try:
        _SHARD = open_shard(STREAM_ROOT, STREAM_NAME, int(STREAM_SHARD_BYTES))
        return _SHARD
    except Exception as ex:
        log.error("datasets.stream.open_shard_failed root=%s name=%s: %s", STREAM_ROOT, STREAM_NAME, ex, exc_info=True)
        _SHARD = None
        return None


def append_row(kind: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Append one dataset row to the canonical stream.

    Row envelope (recommended):
      {
        "conversation_id": "...",
        "trace_id": "...",
        "tool": "...",
        "inputs": {...},
        "outputs": {...},
        "locks": {...},
        "qa": {...},
        "tags": [...],
        "meta": {...}
      }
    """
    sh = _ensure_shard()
    if sh is None:
        return None
    now_ms = int(time.time() * 1000)
    rid = uuid.uuid4().hex
    payload = dict(row or {})
    # Normalize core fields for queryability.
    tags = _norm_tags(payload.get("tags"))
    outputs = _extract_outputs(payload)
    rec: Dict[str, Any] = {
        "t": now_ms,
        "id": rid,
        "kind": str(kind or "event"),
        "conversation_id": payload.get("conversation_id"),
        "trace_id": payload.get("trace_id"),
        "tool": payload.get("tool"),
        "modality": payload.get("modality"),
        "tags": tags,
        "inputs": payload.get("inputs") if isinstance(payload.get("inputs"), dict) else None,
        "outputs": outputs if outputs else None,
        "locks": payload.get("locks") if isinstance(payload.get("locks"), dict) else None,
        "qa": payload.get("qa") if isinstance(payload.get("qa"), dict) else None,
        "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else None,
        "meta": payload.get("meta") if isinstance(payload.get("meta"), dict) else None,
        # Preserve the full original payload for maximal distillation fidelity.
        "payload": payload,
    }
    try:
        _new = append_jsonl(sh, rec)
        # store back
        global _SHARD
        _SHARD = _new
        # Keep a compact meta index for discovery.
        try:
            _update_meta(rec["kind"], rec)
        except Exception:
            log.debug("datasets.stream.update_meta_failed kind=%s", rec.get("kind"), exc_info=True)
        return rec
    except Exception as ex:
        log.error("datasets.stream.append_failed kind=%s: %s", kind, ex, exc_info=True)
        return None


def append_sample(modality: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper for training samples.
    """
    rec = dict(row or {})
    rec.setdefault("modality", str(modality or "").strip().lower())
    # Ensure a few canonical tags.
    tags = _norm_tags(rec.get("tags"))
    tags.append(f"kind:training_sample")
    if rec.get("modality"):
        tags.append(f"modality:{rec.get('modality')}")
    if isinstance(rec.get("tool"), str):
        tags.append(f"tool:{rec.get('tool')}")
    rec["tags"] = sorted(list(dict.fromkeys(tags)))
    return append_row("training_sample", rec)


def append_fact(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper for grounded facts.
    """
    rec = dict(row or {})
    tags = _norm_tags(rec.get("tags"))
    tags.append("kind:fact")
    rec["tags"] = sorted(list(dict.fromkeys(tags)))
    return append_row("fact", rec)


