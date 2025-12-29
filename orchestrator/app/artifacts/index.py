from __future__ import annotations

"""
Canonical artifact index + resolver.

This replaces the older `app/context/index.py` implementation by moving the same
capability under `app/artifacts/` (the correct domain for tracing/distillation).

Responsibilities:
- In-process recent artifact cache per conversation (`conversation_id`)
- Append-only persistent global index (sharded JSONL + index.json)
- Heuristic resolution for "last image", color mentions, stems, etc.
- Optional prompt/text embeddings for retrieval (SentenceTransformers)
"""

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ..json_parser import JSONParser
from .shard import open_shard as _open_shard, append_jsonl as _append_jsonl


_LOG = logging.getLogger(__name__)

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# In-memory per-conversation_id cache (fast-path for "last"/"previous" in an active run).
_CTX: Dict[str, List[Dict[str, Any]]] = {}

# Persistent global index (single system: lives under uploads/artifacts/).
_UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join("/workspace", "uploads"))
_GLOBAL_ROOT = os.getenv("ARTIFACT_INDEX_ROOT", os.path.join(_UPLOAD_DIR, "artifacts", "index"))
_GLOBAL_NAME = os.getenv("ARTIFACT_INDEX_NAME", "global")
_GLOBAL_MAX_BYTES = int(os.getenv("ARTIFACT_INDEX_SHARD_BYTES", os.getenv("ARTIFACT_SHARD_BYTES", "50000000")))
_GLOBAL_SHARD: Optional[dict] = None

# Optional semantic text embedding for prompt/text hints (best-effort).
_TEXT_EMB_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
_TEXT_EMB = None


_COLOR_NAMES: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 20, 60),
    "green": (34, 139, 34),
    "blue": (30, 144, 255),
    "yellow": (250, 250, 70),
    "purple": (128, 0, 128),
    "orange": (255, 140, 0),
    "pink": (255, 105, 180),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}


def _dominant_color(path: str) -> Optional[Tuple[int, int, int]]:
    if Image is None:
        return None
    try:
        with Image.open(path) as im:  # type: ignore
            im = im.convert("RGB")  # type: ignore
            im = im.resize((32, 32))  # type: ignore
            pixels = list(im.getdata())  # type: ignore
            if not pixels:
                return None
            r = sum(p[0] for p in pixels) // len(pixels)
            g = sum(p[1] for p in pixels) // len(pixels)
            b = sum(p[2] for p in pixels) // len(pixels)
            return (int(r), int(g), int(b))
    except Exception as ex:
        _LOG.warning("artifacts.index._dominant_color.error", extra={"path": path}, exc_info=ex)
        return None


def _color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _cos(a, b) -> float:
    try:
        if not a or not b:
            return 0.0
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a))
        db = math.sqrt(sum(y * y for y in b))
        return float(num) / (da * db + 1e-9)
    except Exception as ex:
        _LOG.warning("artifacts.index._cos.error", exc_info=ex)
        return 0.0


def _embed_text(text: str) -> Optional[List[float]]:
    """
    Best-effort text embedding for retrieval. Logs failures; returns None when unavailable.
    """
    global _TEXT_EMB
    if not text:
        return None
    if SentenceTransformer is None:
        return None
    try:
        if _TEXT_EMB is None:
            _TEXT_EMB = SentenceTransformer(_TEXT_EMB_MODEL)
        vec = _TEXT_EMB.encode([text], normalize_embeddings=True)
        if hasattr(vec, "tolist"):
            return vec.tolist()[0]
        if isinstance(vec, list) and vec:
            return vec[0]
        return None
    except Exception as ex:
        _LOG.warning("artifacts.index._embed_text.error", extra={"model": _TEXT_EMB_MODEL}, exc_info=ex)
        return None


def _ensure_global_shard() -> Optional[dict]:
    global _GLOBAL_SHARD
    try:
        if _GLOBAL_SHARD is None:
            _GLOBAL_SHARD = _open_shard(_GLOBAL_ROOT, _GLOBAL_NAME, int(_GLOBAL_MAX_BYTES))
        return _GLOBAL_SHARD
    except Exception as ex:
        _LOG.error(
            "artifacts.index.open_shard_failed",
            extra={"root": _GLOBAL_ROOT, "name": _GLOBAL_NAME, "max_bytes": _GLOBAL_MAX_BYTES},
            exc_info=ex,
        )
        return None


def add_artifact(
    conversation_id: str,
    kind: str,
    path: str,
    url: Optional[str] = None,
    parent: Optional[str] = None,
    tags: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record an artifact into the in-memory conversation_id cache and persistent global index.
    """
    if not isinstance(conversation_id, str) or not conversation_id:
        _LOG.warning("artifacts.index.add_artifact.invalid_conversation_id conversation_id=%r kind=%r path=%r", conversation_id, kind, path)
        return
    # Normalize canonical creation timestamps.
    now_ms = int(time.time() * 1000)
    meta_obj: Dict[str, Any] = dict(meta or {})
    created_ms = meta_obj.get("created_ms")
    if not isinstance(created_ms, int):
        created_at = meta_obj.get("created_at")
        if isinstance(created_at, int) and created_at > 0:
            created_ms = int(created_at) * 1000
        else:
            created_ms = int(now_ms)
    meta_obj.setdefault("created_ms", int(created_ms))
    meta_obj.setdefault("created_at", int(int(created_ms) // 1000))

    base_tags: List[str] = list(tags or [])
    # Add stable, searchable time tags so callers can always find newest artifacts.
    created_tag = f"created_ms:{int(created_ms)}"
    created_at_tag = f"created_at:{int(int(created_ms)//1000)}"
    if created_tag not in base_tags:
        base_tags.append(created_tag)
    if created_at_tag not in base_tags:
        base_tags.append(created_at_tag)

    rec: Dict[str, Any] = {
        "kind": kind,
        "path": path,
        "url": url,
        "parent": parent,
        "tags": base_tags,
        "meta": meta_obj,
    }
    # Optional text embedding from prompt/text.
    try:
        txt = str((rec["meta"].get("prompt") or rec["meta"].get("text") or "")).strip()
        tvec = _embed_text(txt) if txt else None
        if tvec:
            rec.setdefault("emb", {})["text"] = tvec
    except Exception as ex:
        _LOG.warning("artifacts.index.add_artifact.embed_error", extra={"conversation_id": conversation_id, "kind": kind, "path": path}, exc_info=ex)
    # Image color heuristic
    try:
        if isinstance(kind, str) and kind.startswith("image") and isinstance(path, str) and os.path.exists(path):
            dc = _dominant_color(path)
            if dc:
                rec["dominant_rgb"] = dc
    except Exception as ex:
        _LOG.warning("artifacts.index.add_artifact.color_error", extra={"conversation_id": conversation_id, "kind": kind, "path": path}, exc_info=ex)

    _CTX.setdefault(conversation_id, []).append(rec)

    # Persist
    sh = _ensure_global_shard()
    if sh is None:
        return
    row = {
        "ts": int(time.time()),
        "created_ms": int(created_ms),
        "created_at": int(int(created_ms) // 1000),
        "conversation_id": conversation_id,
        "kind": kind,
        "path": path,
        "url": url,
        "parent": parent,
        "tags": rec.get("tags") or [],
        "meta": rec.get("meta") or {},
        "dominant_rgb": rec.get("dominant_rgb"),
        "emb": rec.get("emb") or {},
    }
    try:
        _GLOBAL_SHARD = _append_jsonl(sh, row)
    except Exception as ex:
        _LOG.error(
            "artifacts.index.append_failed",
            extra={"conversation_id": conversation_id, "kind": kind, "path": path, "root": _GLOBAL_ROOT, "name": _GLOBAL_NAME},
            exc_info=ex,
        )


def resolve_reference(conversation_id: str, text: str, kind_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Resolve a reference within a conversation_id session (fast in-memory).
    """
    if not isinstance(conversation_id, str) or not conversation_id:
        return None
    items = list(_CTX.get(conversation_id) or [])
    if not items:
        return None
    if kind_hint:
        items = [it for it in items if str(it.get("kind", "")).startswith(kind_hint)]
    txt = (text or "").lower()
    if not items:
        return None
    # Color mention for images
    for cname, rgb in _COLOR_NAMES.items():
        if cname in txt:
            scored = []
            for it in items:
                dc = it.get("dominant_rgb")
                if dc:
                    scored.append((it, _color_distance(tuple(dc), rgb)))
            if scored:
                scored = sorted(scored, key=lambda x: x[1])
                return scored[0][0]
            break
    # Stems mention for music/audio
    for key in ("drums", "bass", "lead", "pad", "vocal", "voice"):
        if key in txt:
            for it in reversed(items):
                tags = it.get("tags") or []
                if any(tag == f"stem:{key}" for tag in tags):
                    return it
    # "last" / "previous"
    if ("last" in txt) or ("previous" in txt):
        return items[-1]
    if ("one before" in txt) or ("before we" in txt) or ("go back" in txt):
        last = items[-1]
        par = last.get("parent")
        if par:
            for it in reversed(items):
                if it.get("path") == par:
                    return it
    return items[-1]


def list_recent(conversation_id: str, limit: int = 10, kind_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    items = list(_CTX.get(conversation_id) or [])
    if kind_hint:
        items = [it for it in items if str(it.get("kind", "")).startswith(kind_hint)]
    return items[-limit:]


def _tail_read_lines(path: str, limit: int) -> List[str]:
    try:
        lines: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = 8192
            buf = ""
            while size > 0 and len(lines) < limit:
                take = min(chunk, size)
                size -= take
                f.seek(size)
                buf = f.read(take) + buf
                parts = buf.split("\n")
                if len(parts) > 1:
                    lines = parts[-limit:]
                    buf = parts[0]
            if buf:
                lines.insert(0, buf)
        return [ln for ln in lines if ln and ln.strip()]
    except FileNotFoundError:
        return []
    except Exception as ex:
        _LOG.warning("artifacts.index.tail_read_failed", extra={"path": path}, exc_info=ex)
        return []


def resolve_global(text: str, kind_hint: Optional[str] = None, search_limit: int = 500) -> Optional[Dict[str, Any]]:
    """
    Heuristic search over the tail of the global shard parts.
    """
    try:
        # Read from newest shard parts (up to 3 parts).
        idx_path = os.path.join(_GLOBAL_ROOT, f"{_GLOBAL_NAME}.index.json")
        parts: List[str] = []
        parser = JSONParser()
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = parser.parse(f.read(), {})
            if isinstance(idx, dict):
                for part in (idx.get("parts") or [])[-3:]:
                    if isinstance(part, dict) and isinstance(part.get("path"), str):
                        parts.append(os.path.join(_GLOBAL_ROOT, str(part["path"])))
        except FileNotFoundError:
            return None
        except Exception as ex:
            _LOG.warning("artifacts.index.read_index_failed", extra={"index_path": idx_path}, exc_info=ex)
            return None

        lines: List[str] = []
        per = max(50, int(search_limit // max(1, len(parts)))) if parts else search_limit
        for p in reversed(parts):
            lines.extend(_tail_read_lines(p, per))
            if len(lines) >= search_limit:
                lines = lines[-search_limit:]
                break

        txt = (text or "").lower()
        best = None
        best_score = -1.0
        qvec = _embed_text(txt)
        for ln in reversed(lines):
            obj = parser.parse(ln, {})
            if not isinstance(obj, dict):
                continue
            if kind_hint and not str(obj.get("kind", "")).startswith(kind_hint):
                continue
            score = 0.0
            tags = obj.get("tags") or []
            meta = obj.get("meta") or {}
            prompt = str(meta.get("prompt") or meta.get("text") or "").lower()
            for tok in txt.split():
                if tok in prompt:
                    score += 2.0
                if any(tok in str(t).lower() for t in tags):
                    score += 1.0
            # embedding similarity boost
            try:
                if qvec is not None and isinstance(obj.get("emb"), dict) and obj["emb"].get("text"):
                    score += 3.0 * max(0.0, _cos(qvec, obj["emb"]["text"]))
            except Exception as ex:
                _LOG.debug("artifacts.index.emb_similarity_error", extra={"kind": obj.get("kind")}, exc_info=ex)
            # color hint boost
            if str(obj.get("kind", "")).startswith("image") and obj.get("dominant_rgb"):
                for cname in _COLOR_NAMES.keys():
                    if cname in txt:
                        score += 1.0
                        break
            if score > best_score:
                best_score = score
                best = obj
        return best
    except Exception as ex:
        _LOG.error("artifacts.index.resolve_global.error", exc_info=ex)
        return None


_EMOTION_LABELS = [
    "happy",
    "sad",
    "angry",
    "calm",
    "excited",
    "romantic",
    "dark",
    "epic",
    "mysterious",
    "melancholic",
]


def infer_audio_emotion(text_hint: str) -> Optional[str]:
    try:
        qvec = _embed_text(text_hint)
        if qvec is None:
            return None
        best = None
        best_s = -1.0
        for lab in _EMOTION_LABELS:
            lvec = _embed_text(lab)
            if lvec is None:
                continue
            s = _cos(qvec, lvec)
            if s > best_s:
                best_s = s
                best = lab
        return best
    except Exception as ex:
        _LOG.warning("artifacts.index.infer_audio_emotion.error", exc_info=ex)
        return None


