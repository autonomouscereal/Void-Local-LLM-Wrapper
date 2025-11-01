from __future__ import annotations

import hashlib
import time
from typing import List, Dict


def _hash_excerpt(url: str, excerpt: str) -> str:
    return hashlib.sha1((url or "")[:400].encode() + (excerpt or "")[:800].encode()).hexdigest()


def distilled_excerpt(raw: str) -> str:
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    return "\n".join(lines[:2])[:800]


def normalize_sources(sources: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    now = int(time.time())
    for s in (sources or []):
        ex = distilled_excerpt(s.get("text", ""))
        hid = _hash_excerpt(s.get("url", ""), ex)
        rows.append({
            "id": hid,
            "url": s.get("url", ""),
            "title": (s.get("title", "") or "")[:220],
            "excerpt": ex,
            "hash": "sha1:" + hid,
            "fetched_at": now,
            "ts": int(s.get("ts") or now),
        })
    return rows


