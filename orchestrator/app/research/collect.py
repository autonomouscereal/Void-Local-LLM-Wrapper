from __future__ import annotations

from typing import List, Dict, Optional
import time
import os
import json
import urllib.parse
import urllib.request
import asyncio

from ..rag.core import rag_search as _rag_search  # type: ignore


def discover_sources(query: str, scope: str, since: Optional[str], until: Optional[str]) -> List[Dict]:
    """
    Deterministic multi-source discovery using available backends.
    Returns list of { url, title, text, ts }.

    RAG search runs as an async helper; in this synchronous wrapper we bridge
    using asyncio.run and let any event-loop misuse surface as a hard error
    rather than silently swallowing it.
    """
    out: List[Dict] = []
    now = int(time.time())

    # Local RAG hits
    rag = asyncio.run(_rag_search(query, k=8))
    for r in rag or []:
        url = r.get("path") or r.get("url") or ""
        title = r.get("path") or "local"
        text = r.get("chunk") or r.get("text") or ""
        out.append({"url": url, "title": title, "text": text, "ts": now})

    return out


