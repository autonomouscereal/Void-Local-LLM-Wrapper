import os
import asyncio
from typing import Optional, Any, Dict, List
import logging
import traceback

_pool = None


async def get_pg_pool():
    global _pool
    if _pool is not None:
        return _pool
    dsn = os.getenv("PG_DSN") or ""
    if not dsn:
        return None
    try:
        import asyncpg  # type: ignore
    except Exception as e:
        logging.error("asyncpg import failed for PG_DSN=%s: %s\n%s", dsn, e, traceback.format_exc())
        return None
    try:
        _pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)
        return _pool
    except Exception as e:
        logging.error("asyncpg.create_pool failed for PG_DSN=%s: %s\n%s", dsn, e, traceback.format_exc())
        return None


async def ensure_tables():
    pool = await get_pg_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute("""
        create table if not exists tool_schemas (
          tool_name text not null,
          version text not null,
          schema_hash text not null,
          schema_json jsonb not null,
          added_at timestamptz not null default now(),
          primary key (tool_name, version, schema_hash)
        );
        """)
        await conn.execute("""
        create table if not exists tool_patches (
          tool_name text not null,
          version text not null,
          from_hash text not null,
          to_hash text not null,
          patch_id bigserial primary key,
          patch_json jsonb not null,
          created_at timestamptz not null default now()
        );
        """)
        await conn.execute("""
        create table if not exists tool_call_telemetry (
          trace_id text not null,
          step_id text not null,
          tool_name text not null,
          version text not null,
          builder_ok boolean not null,
          repair_rounds int not null default 0,
          retargeted boolean not null default false,
          status_code int not null,
          error_json jsonb,
          final_args jsonb not null,
          created_at timestamptz not null default now()
        );
        """)


