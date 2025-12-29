import os
import hashlib
import json
from typing import Any, Dict, Optional
import time
import logging
import traceback

from void_json.json_parser import JSONParser

from .db import get_pg_pool


def _hash_schema(schema: Dict[str, Any]) -> str:
    compact = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(compact).hexdigest()


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request
    import urllib.error

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        # Critical: do NOT raise — executor must continue even if tool.describe is missing.
        try:
            raw = (e.read() or b"").decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        logging.warning("schema_fetcher._post HTTPError url=%s status=%s body_prefix=%r", url, e.code, (raw or "")[:200])
        return {"ok": False, "error": {"code": "http_error", "status": int(getattr(e, "code", 0) or 0), "message": "tool.describe http error"}, "raw": raw}
    except urllib.error.URLError as e:
        logging.warning("schema_fetcher._post URLError url=%s err=%r", url, e)
        return {"ok": False, "error": {"code": "url_error", "status": 0, "message": str(e)}, "raw": ""}
    try:
        parser = JSONParser()
        obj = parser.parse(raw, {}) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        logging.error("schema_fetcher._post failed for url=%s: %s\n%s", url, e, traceback.format_exc())
        return {}


async def fetch(tool_name: str) -> Dict[str, Any]:
    """Fetch tool describe and cache to Postgres (best-effort)."""
    base = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
    url = base.rstrip("/") + "/tool.describe"
    obj = _post(url=url, payload={"tool_name": tool_name})
    res = (obj or {}).get("result") or {}
    schema = res.get("schema") or {}
    version = res.get("version") or "1"
    schema_hash = res.get("schema_hash") or _hash_schema(schema if isinstance(schema, dict) else {})
    pool = await get_pg_pool()
    if pool is not None and isinstance(schema, dict):
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "insert into tool_schemas (tool_name, version, schema_hash, schema_json) values ($1,$2,$3,$4) on conflict do nothing",
                    tool_name, version, schema_hash, json.dumps(schema),
                )
            except Exception as e:
                logging.error("schema_fetcher.fetch failed to cache schema for tool=%s: %s\n%s", tool_name, e, traceback.format_exc())
    return {"tool_name": tool_name, "version": version, "schema": (schema or {}), "schema_hash": schema_hash}


