from __future__ import annotations

import os
from typing import Optional

import asyncpg  # type: ignore


pg_pool: Optional[asyncpg.pool.Pool] = None


async def get_pg_pool() -> Optional[asyncpg.pool.Pool]:
    global pg_pool
    if pg_pool is not None:
        return pg_pool
    host = os.getenv("POSTGRES_HOST")
    db = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    port = int(os.getenv("POSTGRES_PORT", "5432") or 5432)
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
              run_id      BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
              name        TEXT NOT NULL,
              seed        BIGINT NOT NULL,
              args_json   JSONB NOT NULL,
              result_json JSONB,
              artifact_id BIGINT REFERENCES artifact(id),
              duration_ms INTEGER,
              created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS tool_run_name_idx ON tool_call(run_id, name);")
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
        except Exception:
            pass
        try:
            await conn.execute("CREATE INDEX IF NOT EXISTS rag_docs_embedding_idx ON rag_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
        except Exception:
            try:
                await conn.execute("CREATE INDEX IF NOT EXISTS rag_docs_embedding_hnsw_idx ON rag_docs USING hnsw (embedding vector_l2_ops);")
            except Exception:
                pass
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
    return pg_pool


