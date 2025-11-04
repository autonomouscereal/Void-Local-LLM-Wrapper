from __future__ import annotations

from typing import List, Dict, Optional
import time
import os
import json
import urllib.parse
import urllib.request
import asyncio


def discover_sources(query: str, scope: str, since: Optional[str], until: Optional[str]) -> List[Dict]:
    """
    Deterministic multi-source discovery using available backends.
    Returns list of { url, title, text, ts }.
    """
    out: List[Dict] = []
    now = int(time.time())
    # SERPAPI removed; rely on metasearch and local RAG
    # Local RAG hits — import from rag.core to avoid circular deps
    try:
        from ..rag.core import rag_search as _rag_search  # type: ignore
        try:
            rag = asyncio.run(_rag_search(query, k=8))  # if no running loop
        except RuntimeError:
            # Already in an event loop; skip to avoid nested loop issues
            rag = []
        for r in rag or []:
            url = r.get("path") or r.get("url") or ""
            title = r.get("path") or "local"
            text = r.get("chunk") or r.get("text") or ""
            out.append({"url": url, "title": title, "text": text, "ts": now})
    except Exception:
        pass
    return out


