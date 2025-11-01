from __future__ import annotations

from typing import List, Dict, Optional
import time
from ..main import serpapi_google_search, rag_search


def discover_sources(query: str, scope: str, since: Optional[str], until: Optional[str]) -> List[Dict]:
    """
    Deterministic multi-source discovery using available backends.
    Returns list of { url, title, text, ts }.
    """
    out: List[Dict] = []
    now = int(time.time())
    # SERPAPI (if enabled via env)
    try:
        txt = serpapi_google_search([query], max_results=5)
        if isinstance(txt, str) and txt.strip():
            lines = [ln for ln in txt.splitlines() if ln.strip()]
            for i in range(0, len(lines), 3):
                try:
                    title = lines[i]
                    snippet = lines[i + 1] if i + 1 < len(lines) else ""
                    link = lines[i + 2] if i + 2 < len(lines) else ""
                    if link:
                        out.append({"url": link, "title": title, "text": snippet, "ts": now})
                except Exception:
                    break
    except Exception:
        pass
    # Local RAG hits
    try:
        rag = rag_search(query, k=8)
        for r in rag or []:
            url = r.get("path") or r.get("url") or ""
            title = r.get("path") or "local"
            text = r.get("chunk") or r.get("text") or ""
            out.append({"url": url, "title": title, "text": text, "ts": now})
    except Exception:
        pass
    return out


