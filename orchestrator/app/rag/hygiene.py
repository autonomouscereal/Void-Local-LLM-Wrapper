from __future__ import annotations

import time
import re
import hashlib
import os
from typing import List, Dict


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def _excerpt_hash(text: str) -> str:
    return hashlib.sha1(((text or "")[:1200]).encode("utf-8")).hexdigest()


def rag_filter(chunks: List[Dict], ttl_s: int | None = None) -> List[Dict]:
    """
    chunks: [{"title": str, "text": str, "ts": epoch_seconds, "url": str, ...}, ...]
    Keeps only fresh, non-duplicate items. Sorts newest-first.
    Duplicate key = (normalized title, excerpt hash).
    """
    if ttl_s is None:
        try:
            ttl_s = int(os.getenv("RAG_TTL_SECONDS", "3600"))
        except Exception:
            ttl_s = 3600
    now = time.time()
    seen = set()
    fresh: List[Dict] = []
    for c in chunks or []:
        if not isinstance(c, dict):
            continue
        try:
            ts = float(c.get("ts") or 0)
        except Exception:
            ts = 0.0
        if ts and (now - ts) > float(ttl_s):
            continue
        key = (_norm_title(c.get("title", "")), _excerpt_hash(c.get("text", "")))
        if key in seen:
            continue
        seen.add(key)
        fresh.append(c)
    fresh.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return fresh


def evidence_binding_footer(distilled_chunks: List[Dict]) -> str:
    """
    Tiny footer to nudge grounding—non-security. Include 1–2 lines per source, newest-first.
    """
    try:
        max_items = int(os.getenv("RAG_EVIDENCE_MAX", "6"))
    except Exception:
        max_items = 6
    lines = []
    for c in (distilled_chunks or [])[: max(1, max_items)]:
        title = c.get("title") or c.get("url") or "source"
        quote = (c.get("text") or "").strip().splitlines()[0][:220]
        if quote:
            lines.append(f"- {title}: {quote}")
        else:
            lines.append(f"- {title}")
    if not lines:
        return ""
    return "\n\n[Evidence]\n" + "\n".join(lines) + "\n\n(When making factual claims, ground them in the Evidence above or say you're unsure.)"


