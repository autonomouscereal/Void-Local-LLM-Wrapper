from __future__ import annotations

from typing import List, Dict, Optional
import time


def discover_sources(query: str, scope: str, since: Optional[str], until: Optional[str]) -> List[Dict]:
    """
    Stub implementation. Wire in your actual search stack here (web/metasearch/RAG).
    Returns list of { url, title, text, ts }.
    """
    # Placeholder: return empty, caller handles downstream
    now = time.time()
    return [{"url": "", "title": "", "text": "", "ts": now}][:0]


