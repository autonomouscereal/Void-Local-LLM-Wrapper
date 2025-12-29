from __future__ import annotations

import json
import logging
from typing import Optional, Dict, Any

from .pool import get_pg_pool

log = logging.getLogger(__name__)

async def db_insert_run(*, trace_id: str, mode: str, seed: int, pack_hash: Optional[str], request_json: Dict[str, Any]) -> Optional[int]:
    try:
        pool = await get_pg_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO run(trace_id, mode, seed, pack_hash, request_json) VALUES($1,$2,$3,$4,$5) ON CONFLICT (trace_id) DO UPDATE SET mode=EXCLUDED.mode RETURNING id",
                trace_id, mode, int(seed), pack_hash, json.dumps(request_json, ensure_ascii=False),
            )
            return int(row[0]) if row else None
    except Exception:
        # Tracing must never break the request path, but silent failures make debugging impossible.
        log.warning(f"db.tracing.insert_run_failed trace_id={trace_id!r} mode={mode!r}", exc_info=True)
        return None


async def db_update_run_response(*, trace_id: str, response_json: Dict[str, Any], metrics_json: Dict[str, Any]) -> None:
    if not isinstance(trace_id, str) or not trace_id:
        return
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute("UPDATE run SET response_json=$1, metrics_json=$2 WHERE trace_id=$3", json.dumps(response_json, ensure_ascii=False), json.dumps(metrics_json, ensure_ascii=False), trace_id)
    except Exception:
        log.warning(f"db.tracing.update_run_response_failed trace_id={trace_id!r}", exc_info=True)
        return


async def db_insert_icw_log(*, trace_id: str, pack_hash: Optional[str], budget_tokens: int, scores_json: Dict[str, Any]) -> None:
    if not (isinstance(trace_id, str) and trace_id and pack_hash):
        return
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO icw_log(run_id, pack_hash, budget_tokens, scores_json) VALUES((SELECT id FROM run WHERE trace_id=$1),$2,$3,$4)",
                trace_id, pack_hash, int(budget_tokens), json.dumps(scores_json, ensure_ascii=False),
            )
    except Exception:
        log.warning(f"db.tracing.insert_icw_log_failed trace_id={trace_id!r} pack_hash={pack_hash!r}", exc_info=True)
        return


async def db_insert_tool_call(*, trace_id: str, tool_name: str, seed: int, args_json: Dict[str, Any], result_json: Optional[Dict[str, Any]] = None, duration_ms: Optional[int] = None, artifact_id: Optional[Any] = None) -> None:
    """
    Insert tool call into database. artifact_id accepts any type (str, int, etc.) and converts to int for DB only if valid numeric value.
    String identifiers like "master" or "reel" are not inserted as DB artifact_id (they remain as string identifiers in application layer).
    """
    if not isinstance(trace_id, str) or not trace_id:
        return
    # Convert artifact_id to int for DB only if it's a valid numeric value (database artifact.id)
    # String identifiers like "master", "reel", or filenames are not DB artifact IDs
    artifact_id_db = None
    if artifact_id is not None:
        if isinstance(artifact_id, int):
            artifact_id_db = artifact_id
        elif isinstance(artifact_id, str):
            # Only convert numeric strings (database IDs), not string identifiers
            if artifact_id.isdigit():
                try:
                    artifact_id_db = int(artifact_id)
                except (ValueError, TypeError):
                    artifact_id_db = None
            # String identifiers like "master", "reel", filenames are not DB artifact IDs
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO tool_call(trace_id, tool_name, seed, args_json, result_json, artifact_id, duration_ms) VALUES($1,$2,$3,$4,$5,$6,$7)",
                trace_id, tool_name, int(seed), json.dumps(args_json, ensure_ascii=False), json.dumps(result_json, ensure_ascii=False) if isinstance(result_json, dict) else None, artifact_id_db, int(duration_ms) if duration_ms is not None else None,
            )
    except Exception:
        log.warning(f"db.tracing.insert_tool_call_failed trace_id={trace_id!r} tool_name={tool_name!r}", exc_info=True)
        return


