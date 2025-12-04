from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..db.pool import get_pg_pool
from ..json_parser import JSONParser

_LOCK_DIR_ENV = "LOCKS_DIR"


def locks_root(upload_dir: str) -> str:
    base = os.getenv(_LOCK_DIR_ENV)
    if isinstance(base, str) and base.strip():
        root = os.path.abspath(base.strip())
    else:
        root = os.path.join(upload_dir, "locks")
    os.makedirs(root, exist_ok=True)
    return root


def bundle_path(root: str, character_id: str) -> str:
    safe_id = "".join(c for c in character_id if c.isalnum() or c in ("-", "_"))
    if not safe_id:
        safe_id = "anon"
    return os.path.join(root, safe_id)


async def upsert_lock_bundle(character_id: str, bundle: Dict[str, Any]) -> None:
    pool = await get_pg_pool()
    if pool is None:
        return
    import json
    payload = json.dumps(bundle, ensure_ascii=False)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO lock_bundle (character_id, bundle_json, created_at, updated_at)
            VALUES ($1, $2::jsonb, now(), now())
            ON CONFLICT (character_id) DO UPDATE
            SET bundle_json = EXCLUDED.bundle_json, updated_at = now()
            """,
            character_id,
            payload,
        )


async def get_lock_bundle(character_id: str) -> Optional[Dict[str, Any]]:
    pool = await get_pg_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT bundle_json FROM lock_bundle WHERE character_id = $1",
            character_id,
        )
        if row is None:
            return None
        val = row[0]
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            parser = JSONParser()
            # Lock bundles are arbitrary dicts; coerce to generic mapping.
            parsed = parser.parse_superset(val, dict)["coerced"]
            return parsed if isinstance(parsed, dict) else None
        return None

