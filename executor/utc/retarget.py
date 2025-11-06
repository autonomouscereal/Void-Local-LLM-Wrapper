import json
from typing import Any, Dict, Optional
from ..app.main import ORCHESTRATOR_BASE_URL


def _get(url: str) -> Dict[str, Any]:
    import urllib.request
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def find_candidate(name: str) -> Optional[Dict[str, Any]]:
    """Pick nearest by name/kind. Minimal heuristic: same prefix before '.' or exact kind match."""
    lst = _get(ORCHESTRATOR_BASE_URL.rstrip("/") + "/tool.list")
    tools = ((lst or {}).get("result") or {}).get("tools") or []
    if not tools:
        return None
    base = name.split(".")[0]
    exact_kind = None
    for t in tools:
        if t.get("name") == name:
            exact_kind = t.get("kind")
            break
    # prefer same base prefix, then same kind
    for t in tools:
        if t.get("name", "").split(".")[0] == base and t.get("name") != name:
            return t
    if exact_kind:
        for t in tools:
            if t.get("kind") == exact_kind and t.get("name") != name:
                return t
    return None


