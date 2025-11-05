from __future__ import annotations

import os
import threading
import json
import time
from typing import Dict, Any, List, Optional, Tuple


_LOCK = threading.RLock()
_CTX: Dict[str, List[Dict[str, Any]]] = {}
_GLOBAL_PATH = os.getenv("ARTIFACTS_INDEX_PATH", os.path.join("/workspace", "uploads", "artifacts", "index.jsonl"))
_TEXT_EMB_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
_TEXT_EMB = None


_COLOR_NAMES = {
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
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(path) as im:  # type: ignore
            im = im.convert("RGB")  # type: ignore
            im = im.resize((32, 32))  # type: ignore
            pixels = list(im.getdata())  # type: ignore
            # Simple average
            r = sum(p[0] for p in pixels) // len(pixels)
            g = sum(p[1] for p in pixels) // len(pixels)
            b = sum(p[2] for p in pixels) // len(pixels)
            return (int(r), int(g), int(b))
    except Exception:
        return None


def _color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5


def _cos(a, b) -> float:
    try:
        if not a or not b:
            return 0.0
        import math
        num = sum(x*y for x,y in zip(a,b))
        da = math.sqrt(sum(x*x for x in a))
        db = math.sqrt(sum(y*y for y in b))
        return float(num) / (da*db + 1e-9)
    except Exception:
        return 0.0


def _embed_text(text: str) -> Optional[List[float]]:
    global _TEXT_EMB
    try:
        if not text:
            return None
        if _TEXT_EMB is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _TEXT_EMB = SentenceTransformer(_TEXT_EMB_MODEL)
        vec = _TEXT_EMB.encode([text], normalize_embeddings=True)
        if hasattr(vec, "tolist"):
            return vec.tolist()[0]
        if isinstance(vec, list) and vec:
            return vec[0]
    except Exception:
        return None
    return None


def add_artifact(cid: str, kind: str, path: str, url: Optional[str] = None, parent: Optional[str] = None, tags: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> None:
    if not isinstance(cid, str) or not cid:
        return
    rec = {
        "kind": kind,
        "path": path,
        "url": url,
        "parent": parent,
        "tags": list(tags or []),
        "meta": dict(meta or {}),
    }
    # Optional text embedding from prompt/text
    try:
        txt = str((rec["meta"].get("prompt") or rec["meta"].get("text") or "")).strip()
        tvec = _embed_text(txt) if txt else None
        if tvec:
            rec.setdefault("emb", {})["text"] = tvec
    except Exception:
        pass
    if kind.startswith("image") and os.path.exists(path):
        dc = _dominant_color(path)
        if dc:
            rec["dominant_rgb"] = dc
    with _LOCK:
        _CTX.setdefault(cid, []).append(rec)
    # Append to global NDJSON for cross-conversation memory
    try:
        os.makedirs(os.path.dirname(_GLOBAL_PATH), exist_ok=True)
        row = {
            "ts": int(time.time()),
            "cid": cid,
            "kind": kind,
            "path": path,
            "url": url,
            "parent": parent,
            "tags": rec.get("tags") or [],
            "meta": rec.get("meta") or {},
            "dominant_rgb": rec.get("dominant_rgb"),
            "emb": rec.get("emb") or {},
        }
        with open(_GLOBAL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def resolve_reference(cid: str, text: str, kind_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(cid, str) or not cid:
        return None
    with _LOCK:
        items = list(_CTX.get(cid) or [])
    if not items:
        return None
    # Filter by kind hint if provided
    if kind_hint:
        items = [it for it in items if it.get("kind", "").startswith(kind_hint)]
    txt = (text or "").lower()
    if not items:
        return None
    # Heuristic 1: color mention for images
    for cname, rgb in _COLOR_NAMES.items():
        if cname in txt:
            scored = []
            for it in items:
                dc = it.get("dominant_rgb")
                if dc:
                    scored.append((it, _color_distance(tuple(dc), rgb)))
            if scored:
                scored.sort(key=lambda x: x[1])
                return scored[0][0]
            break
    # Heuristic 2: stems mention for music/audio
    for key in ("drums", "bass", "lead", "pad", "vocal", "voice"):
        if key in txt:
            for it in reversed(items):
                tags = it.get("tags") or []
                if any(tag == f"stem:{key}" for tag in tags):
                    return it
    # Heuristic 3: "last" / "previous" / "before" selections
    if ("last" in txt) or ("previous" in txt):
        return items[-1]
    if ("one before" in txt) or ("before we" in txt) or ("go back" in txt):
        # Parent of last if available
        last = items[-1]
        par = last.get("parent")
        if par:
            for it in reversed(items):
                if it.get("path") == par:
                    return it
    # Fallback: just return the most recent of the hinted kind
    return items[-1]


def list_recent(cid: str, limit: int = 10, kind_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    with _LOCK:
        items = list(_CTX.get(cid) or [])
    if kind_hint:
        items = [it for it in items if it.get("kind", "").startswith(kind_hint)]
    return items[-limit:]


def resolve_global(text: str, kind_hint: Optional[str] = None, search_limit: int = 500) -> Optional[Dict[str, Any]]:
    # Lightweight heuristic search over the tail of the global index
    try:
        if not os.path.exists(_GLOBAL_PATH):
            return None
        # Read last N lines efficiently
        lines: List[str] = []
        with open(_GLOBAL_PATH, "r", encoding="utf-8") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                chunk = 8192
                buf = ""
                while size > 0 and len(lines) < search_limit:
                    take = min(chunk, size)
                    size -= take
                    f.seek(size)
                    buf = f.read(take) + buf
                    parts = buf.split("\n")
                    if len(parts) > 1:
                        lines = parts[-search_limit:]
                        buf = parts[0]
                if buf:
                    lines.insert(0, buf)
            except Exception:
                f.seek(0)
                lines = f.readlines()[-search_limit:]
        # Score
        txt = (text or "").lower()
        best = None
        best_score = -1.0
        qvec = _embed_text(txt)
        for ln in reversed(lines):
            ln = ln.strip()
            if not ln:
                continue
            try:
                from ..jsonio.helpers import parse_json_text as _parse_json_text
                obj = _parse_json_text(ln, {})
            except Exception:
                continue
            if kind_hint and not str(obj.get("kind", "")).startswith(kind_hint):
                continue
            score = 0.0
            tags = obj.get("tags") or []
            meta = obj.get("meta") or {}
            prompt = str(meta.get("prompt") or meta.get("text") or "").lower()
            # simple token overlap
            for tok in txt.split():
                if tok in prompt:
                    score += 2.0
                if any(tok in str(t).lower() for t in tags):
                    score += 1.0
            # embedding similarity boost
            try:
                if qvec is not None and isinstance(obj.get("emb"), dict) and obj["emb"].get("text"):
                    score += 3.0 * max(0.0, _cos(qvec, obj["emb"]["text"]))
            except Exception:
                pass
            # color heuristic for images
            if obj.get("kind", "").startswith("image"):
                for cname, rgb in _COLOR_NAMES.items():
                    if cname in txt and obj.get("dominant_rgb"):
                        score += 1.0
                        break
            if score > best_score:
                best_score = score
                best = obj
        return best
    except Exception:
        return None


_EMOTION_LABELS = [
    "happy", "sad", "angry", "calm", "excited", "romantic", "dark", "epic", "mysterious", "melancholic",
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
    except Exception:
        return None


