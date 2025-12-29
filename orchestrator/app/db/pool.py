from __future__ import annotations

import os
from typing import Optional

import asyncpg  # type: ignore
import logging

log = logging.getLogger(__name__)


pg_pool: Optional[asyncpg.pool.Pool] = None


async def get_pg_pool() -> Optional[asyncpg.pool.Pool]:
    global pg_pool
    if pg_pool is not None:
        return pg_pool
    host = os.getenv("POSTGRES_HOST")
    db = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    # Defensive: env parsing must never raise and break DB init.
    port_raw = os.getenv("POSTGRES_PORT", "5432")
    try:
        port = int(str(port_raw).strip() or "5432")
    except Exception as exc:
        log.warning("db.pool: bad POSTGRES_PORT=%r; defaulting to 5432", port_raw, exc_info=True)
        port = 5432
    if not (host and db and user and password):
        return None
    pg_pool = await asyncpg.create_pool(
        user=user,
        password=password,
        database=db,
        host=host,
        port=port,
        min_size=1,
        max_size=10,
    )
    async with pg_pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Orchestrator-owned conversation registry:
        # - External callers may *suggest* a conversation_id, but only the orchestrator may mint and authorize it.
        # - We keep a lightweight table so we can reject/overwrite unknown conversation_id values deterministically.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orc_conversation (
              conversation_id TEXT PRIMARY KEY,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS orc_conversation_updated_idx ON orc_conversation(updated_at DESC);")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run (
              id            BIGSERIAL PRIMARY KEY,
              trace_id      TEXT UNIQUE NOT NULL,
              workspace     TEXT NOT NULL DEFAULT 'default',
              mode          TEXT NOT NULL,
              seed          BIGINT NOT NULL,
              pack_hash     TEXT,
              request_json  JSONB NOT NULL,
              response_json JSONB,
              metrics_json  JSONB,
              created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS run_mode_created_idx ON run(mode, created_at DESC);")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact (
              id          BIGSERIAL PRIMARY KEY,
              sha256      TEXT NOT NULL UNIQUE,
              uri         TEXT NOT NULL,
              kind        TEXT NOT NULL,
              bytes       BIGINT,
              meta_json   JSONB,
              created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_call (
              id          BIGSERIAL PRIMARY KEY,
              trace_id    TEXT,
              tool_name   TEXT NOT NULL,
              seed        BIGINT NOT NULL,
              args_json   JSONB NOT NULL,
              result_json JSONB,
              artifact_id BIGINT REFERENCES artifact(id),
              duration_ms INTEGER,
              created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        # Don't create index here - migrations will handle it
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS icw_log (
              id            BIGSERIAL PRIMARY KEY,
              run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
              pack_hash     TEXT NOT NULL,
              budget_tokens INTEGER NOT NULL,
              scores_json   JSONB,
              created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        # Teacher traces → distill sets (previously written by the removed `services/teacher` container).
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS teacher_trace (
              id            BIGSERIAL PRIMARY KEY,
              run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
              trace_line    JSONB NOT NULL,
              created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS distill_sft (
              id            BIGSERIAL PRIMARY KEY,
              run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
              sample_json   JSONB NOT NULL,
              created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS distill_dpo (
              id            BIGSERIAL PRIMARY KEY,
              run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
              pair_json     JSONB NOT NULL,
              created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS distill_toolpolicy (
              id            BIGSERIAL PRIMARY KEY,
              run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
              policy_json   JSONB NOT NULL,
              created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_docs (
              id BIGSERIAL PRIMARY KEY,
              path TEXT,
              chunk TEXT,
              embedding vector(1024)
            );
            """
        )
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as ex:
            # Non-fatal: vector extension may not be available in all Postgres builds.
            log.warning("db.pool: failed to create EXTENSION vector: %s", ex, exc_info=True)
        try:
            await conn.execute("CREATE INDEX IF NOT EXISTS rag_docs_embedding_idx ON rag_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
        except Exception as ex:
            log.warning("db.pool: failed to create ivfflat index (will try hnsw): %s", ex, exc_info=True)
            try:
                await conn.execute("CREATE INDEX IF NOT EXISTS rag_docs_embedding_hnsw_idx ON rag_docs USING hnsw (embedding vector_l2_ops);")
            except Exception as ex2:
                log.warning("db.pool: failed to create hnsw index: %s", ex2, exc_info=True)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              prompt_id TEXT,
              status TEXT,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW(),
              workflow JSONB,
              result JSONB,
              error TEXT
            );
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS jobs_status_updated_idx ON jobs (status, updated_at DESC);")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_checkpoints (
              id BIGSERIAL PRIMARY KEY,
              job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              data JSONB
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS films (
              id TEXT PRIMARY KEY,
              title TEXT,
              synopsis TEXT,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW(),
              metadata JSONB
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
              id TEXT PRIMARY KEY,
              film_id TEXT REFERENCES films(id) ON DELETE CASCADE,
              name TEXT,
              description TEXT,
              reference_data JSONB,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scenes (
              id TEXT PRIMARY KEY,
              film_id TEXT REFERENCES films(id) ON DELETE CASCADE,
              index_num INT,
              prompt TEXT,
              plan JSONB,
              status TEXT,
              job_id TEXT,
              assets JSONB,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lock_bundle (
              character_id TEXT PRIMARY KEY,
              bundle_json JSONB NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        # Run migrations AFTER all tables are created to ensure schema is up to date
        await _run_migrations(conn)
        # Create indexes AFTER migrations (migrations may have changed column names)
        await _create_indexes_after_migrations(conn)
    return pg_pool


async def _create_indexes_after_migrations(conn: asyncpg.Connection) -> None:
    """
    Create indexes that depend on migrations having run (e.g., trace_id column).
    """
    try:
        # Check if trace_id exists before creating index
        col_check = await conn.fetchval("""
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'tool_call' AND column_name = 'trace_id'
        """)
        if col_check:
            await conn.execute("CREATE INDEX IF NOT EXISTS tool_run_name_idx ON tool_call(trace_id, tool_name);")
        else:
            # Fallback to run_id if trace_id doesn't exist (shouldn't happen after migrations, but be safe)
            col_check_run_id = await conn.fetchval("""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'tool_call' AND column_name = 'run_id'
            """)
            if col_check_run_id:
                await conn.execute("CREATE INDEX IF NOT EXISTS tool_run_name_idx ON tool_call(run_id, tool_name);")
            else:
                log.warning("db.pool: neither trace_id nor run_id found in tool_call table, skipping index creation")
    except Exception as ex:
        log.warning(f"db.pool: failed to create tool_run_name_idx: {ex}", exc_info=True)


async def _run_migrations(conn: asyncpg.Connection) -> None:
    """
    Run database migrations in order. Best-effort; failures are logged but don't break startup.
    """
    try:
        migrations_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "db",
            "migrations",
        )
        if not os.path.isdir(migrations_dir):
            return
        # Get all migration files and sort them by name (0001, 0002, etc.)
        migration_files = sorted([f for f in os.listdir(migrations_dir) if f.endswith('.sql')])
        if not migration_files:
            return
        for migration_file in migration_files:
            migration_path = os.path.join(migrations_dir, migration_file)
            try:
                with open(migration_path, "r", encoding="utf-8") as f:
                    sql = f.read()
                if sql.strip():
                    # Execute the whole SQL (migrations may contain DO blocks and multi-statement SQL)
                    await conn.execute(sql)
            except Exception as ex:
                # Log but continue - some migrations may fail if already applied
                log.warning(f"db.pool: migration {migration_file} failed (may already be applied): {ex}", exc_info=True)
    except Exception as ex:
        log.warning(f"db.pool: migration runner failed: {ex}", exc_info=True)


