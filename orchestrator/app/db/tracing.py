from __future__ import annotations

import json
from typing import Optional, Dict, Any

from .pool import get_pg_pool


async def db_insert_run(trace_id: str, mode: str, seed: int, pack_hash: Optional[str], request_json: Dict[str, Any]) -> Optional[int]:
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
        return None


async def db_update_run_response(run_id: Optional[int], response_json: Dict[str, Any], metrics_json: Dict[str, Any]) -> None:
    if not run_id:
        return
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute("UPDATE run SET response_json=$1, metrics_json=$2 WHERE id=$3", json.dumps(response_json, ensure_ascii=False), json.dumps(metrics_json, ensure_ascii=False), int(run_id))
    except Exception:
        return


async def db_insert_icw_log(run_id: Optional[int], pack_hash: Optional[str], budget_tokens: int, scores_json: Dict[str, Any]) -> None:
    if not (run_id and pack_hash):
        return
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO icw_log(run_id, pack_hash, budget_tokens, scores_json) VALUES($1,$2,$3,$4)",
                int(run_id), pack_hash, int(budget_tokens), json.dumps(scores_json, ensure_ascii=False),
            )
    except Exception:
        return


async def db_insert_tool_call(run_id: Optional[int], name: str, seed: int, args_json: Dict[str, Any], result_json: Optional[Dict[str, Any]], duration_ms: Optional[int] = None) -> None:
    try:
        pool = await get_pg_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO tool_call(run_id, name, seed, args_json, result_json, duration_ms) VALUES($1,$2,$3,$4,$5,$6)",
                int(run_id) if run_id else None, name, int(seed), json.dumps(args_json, ensure_ascii=False), json.dumps(result_json, ensure_ascii=False) if isinstance(result_json, dict) else None, int(duration_ms) if duration_ms is not None else None,
            )
    except Exception:
        return


