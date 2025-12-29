from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import asyncpg


_pool: Optional[asyncpg.pool.Pool] = None


async def _ensure_base_schema(pool: asyncpg.pool.Pool) -> None:
    """
    Best-effort bootstrap for the core Postgres schema.

    This executes all bundled SQL migrations in order (0001_init.sql, 0002_*.sql, etc.)
    against the current database connection. The files are intentionally plain SQL with
    IF NOT EXISTS guards so they can be safely re-run.
    """
    try:
        migrations_dir = os.path.join(
            os.path.dirname(__file__),
            "migrations",
        )
        if not os.path.isdir(migrations_dir):
            return
        # Get all migration files and sort them by name (0001, 0002, etc.)
        migration_files = sorted([f for f in os.listdir(migrations_dir) if f.endswith('.sql')])
        if not migration_files:
            return
        async with pool.acquire() as conn:
            for migration_file in migration_files:
                migration_path = os.path.join(migrations_dir, migration_file)
                try:
                    with open(migration_path, "r", encoding="utf-8") as f:
                        sql = f.read()
                    if sql.strip():
                        await conn.execute(sql)
                except Exception as ex:
                    # Log but continue - some migrations may fail if already applied
                    import logging
                    logging.warning(f"db.core: migration {migration_file} failed (may already be applied): {ex}", exc_info=True)
    except Exception:
        # Schema bootstrap is best-effort; never break the caller on failure.
        return


async def get_pool() -> Optional[asyncpg.pool.Pool]:
    global _pool
    if _pool is not None:
        return _pool
    host = os.getenv("POSTGRES_HOST")
    db = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    if not (host and db and user and password):
        return None
    _pool = await asyncpg.create_pool(
        host=host,
        port=port,
        user=user,
        password=password,
        database=db,
        min_size=1,
        max_size=10,
    )
    # Ensure the core schema (run / artifact / film / teacher / ablation tables)
    # is present for services that depend on db.core.
    await _ensure_base_schema(_pool)
    return _pool


class Tx:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def __aenter__(self):
        self.tr = self.conn.transaction()
        await self.tr.start()
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        if exc:
            await self.tr.rollback()
        else:
            await self.tr.commit()


async def with_tx() -> Tx:
    pool = await get_pool()
    conn = await pool.acquire()
    return Tx(conn)


async def fetchrow(sql: str, *args) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        return await conn.fetchrow(sql, *args)


async def fetch(sql: str, *args) -> List[asyncpg.Record]:
    pool = await get_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
        return list(rows)


async def execute(sql: str, *args) -> str:
    pool = await get_pool()
    if pool is None:
        return ""
    async with pool.acquire() as conn:
        return await conn.execute(sql, *args)


async def executemany(sql: str, seq_of_params: List[Tuple[Any, ...]]) -> None:
    pool = await get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.executemany(sql, seq_of_params)


