from __future__ import annotations

import os
import requests
from typing import Any, Dict, List


def searxng_search(query: str, base_url: str | None = None, timeout_s: float = 20.0) -> Dict[str, Any]:
    """
    Query SearXNG and return the JSON response.

    Host-network setup note:
    - base_url should match your SEARXNG_PORT (default http://localhost:8090).
    """
    if base_url is None:
        base_url = os.environ.get("SEARXNG_URL", "http://localhost:8090")

    url = base_url.rstrip("/") + "/search"
    r = requests.get(
        url,
        params={"q": query, "format": "json"},
        timeout=timeout_s,
        headers={"User-Agent": "searxng-client/1.0"},
    )
    r.raise_for_status()
    return r.json()


def top_results(data: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """Extract top N results from SearXNG response."""
    return list(data.get("results", []))[:limit]
