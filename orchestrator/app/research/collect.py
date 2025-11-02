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
    # SERPAPI (if enabled via env) — synchronous minimal client to avoid circular imports
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if api_key:
            params = {
                "engine": "google",
                "q": query,
                "num": 5,
                "api_key": api_key,
            }
            url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for item in (data.get("organic_results") or [])[:5]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                if link:
                    out.append({"url": link, "title": title, "text": snippet, "ts": now})
    except Exception:
        pass
    # Local RAG hits — call async function from main lazily to avoid circular import at module import time
    try:
        from ..main import rag_search as _rag_search  # type: ignore
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


