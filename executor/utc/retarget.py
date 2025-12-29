import os
import json
from typing import Any, Dict, Optional
import urllib.request
import urllib.error
import re

from void_json.json_parser import JSONParser
from void_envelopes import _build_error_envelope


def _get(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            parser = JSONParser()
            obj = parser.parse(raw, {}) or {}
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    except urllib.error.HTTPError as e:
        return _build_error_envelope(
            code="not_supported",
            message=f"GET {url} returned HTTPError",
            trace_id="",
            conversation_id="",
            status=int(getattr(e, "code", 0) or 0),
            details={"url": url},
        )


def find_candidate(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort tool name discovery.

    This used to be disabled when orchestrator did not expose /tool.list.
    Orchestrator now provides:
      - POST /tool.list (legacy, minimal)
      - GET  /tool.list (richer)

    We use GET for simplicity and to avoid adding a POST helper here.
    """
    nm = (tool_name or "").strip()
    if not nm:
        return None
    base = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
    url = base.rstrip("/") + "/tool.list"
    env = _get(url)
    if not isinstance(env, dict) or env.get("ok") is not True:
        return None
    result = env.get("result") if isinstance(env.get("result"), dict) else {}
    tools = result.get("tools") if isinstance(result.get("tools"), list) else []

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())

    target = _norm(nm)
    if not target:
        return None

    # Prefer exact match; otherwise choose the closest normalized match.
    best: Optional[Dict[str, Any]] = None
    for t in tools:
        if not isinstance(t, dict):
            continue
        tool_name = t.get("tool_name") or t.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        if tool_name == nm:
            return {"tool_name": tool_name, "version": t.get("version"), "kind": t.get("kind")}
        if _norm(tool_name) == target:
            best = {"tool_name": tool_name, "version": t.get("version"), "kind": t.get("kind")}
            break
    return best


