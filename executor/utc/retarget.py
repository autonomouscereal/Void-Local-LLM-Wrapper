import os
import json
from typing import Any, Dict, Optional
import urllib.request
import urllib.error

from void_json.json_parser import JSONParser


def _get(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            parser = JSONParser()
            sup = parser.parse_superset(raw, {})
            obj = sup.get("coerced") or {}
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    except urllib.error.HTTPError as e:
        return {"schema_version": 1, "ok": False, "code": "not_supported", "status": e.code}


def find_candidate(name: str) -> Optional[Dict[str, Any]]:
    # Disabled per canonical route set; orchestrator does not expose /tool.list
    return None


