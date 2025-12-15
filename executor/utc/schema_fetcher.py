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


def _get(url: str) -> Dict[str, Any]:
    import urllib.request

    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        parser = JSONParser()
        obj = parser.parse(raw, {}) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        logging.error("schema_fetcher._get failed for url=%s: %s\n%s", url, e, traceback.format_exc())
        return {}


async def fetch(name: str) -> Dict[str, Any]:
    """Fetch tool describe and cache to Postgres (best-effort)."""
    base = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
    url = base.rstrip("/") + f"/tool.describe?name={name}"
    obj = _get(url)
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
                    name, version, schema_hash, json.dumps(schema),
                )
            except Exception as e:
                logging.error("schema_fetcher.fetch failed to cache schema for tool=%s: %s\n%s", name, e, traceback.format_exc())
    return {"name": name, "version": version, "schema": (schema or {}), "schema_hash": schema_hash}


