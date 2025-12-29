-- 0001_init.sql — PostgreSQL schema (no ORM). JSONB everywhere; deterministic seeds; content-addressed artifacts.

-- projects / runs
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
CREATE INDEX IF NOT EXISTS run_mode_created_idx ON run(mode, created_at DESC);

-- artifacts (videos, images, audio, packs, json manifests)
CREATE TABLE IF NOT EXISTS artifact (
  id          BIGSERIAL PRIMARY KEY,
  sha256      TEXT NOT NULL UNIQUE,
  uri         TEXT NOT NULL,
  kind        TEXT NOT NULL,
  bytes       BIGINT,
  meta_json   JSONB,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- tools executed within a run
CREATE TABLE IF NOT EXISTS tool_call (
  id          BIGSERIAL PRIMARY KEY,
  trace_id    TEXT NOT NULL,
  tool_name   TEXT NOT NULL,
  seed        BIGINT NOT NULL,
  args_json   JSONB NOT NULL,
  result_json JSONB,
  artifact_id BIGINT REFERENCES artifact(id),
  duration_ms INTEGER,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tool_run_name_idx ON tool_call(trace_id, tool_name);

-- ICW log (inline pack)
CREATE TABLE IF NOT EXISTS icw_log (
  id            BIGSERIAL PRIMARY KEY,
  run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  pack_hash     TEXT NOT NULL,
  budget_tokens INTEGER NOT NULL,
  scores_json   JSONB,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Film 2.0 entities
CREATE TABLE IF NOT EXISTS film_project (
  id            BIGSERIAL PRIMARY KEY,
  project_uid   TEXT NOT NULL UNIQUE,
  seed          BIGINT NOT NULL,
  title         TEXT,
  duration_s    INTEGER NOT NULL,
  config_json   JSONB NOT NULL,
  state         TEXT NOT NULL DEFAULT 'init',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS film_scene (
  id            BIGSERIAL PRIMARY KEY,
  project_id    BIGINT NOT NULL REFERENCES film_project(id) ON DELETE CASCADE,
  scene_uid     TEXT NOT NULL,
  meta_json     JSONB NOT NULL,
  UNIQUE(project_id, scene_uid)
);

CREATE TABLE IF NOT EXISTS film_shot (
  id            BIGSERIAL PRIMARY KEY,
  project_id    BIGINT NOT NULL REFERENCES film_project(id) ON DELETE CASCADE,
  shot_uid      TEXT NOT NULL,
  scene_uid     TEXT NOT NULL,
  dsl_json      JSONB NOT NULL,
  seeds_json    JSONB NOT NULL,
  status        TEXT NOT NULL DEFAULT 'planned',
  artifacts     JSONB,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(project_id, shot_uid)
);
CREATE INDEX IF NOT EXISTS film_shot_status_idx ON film_shot(project_id, status);

CREATE TABLE IF NOT EXISTS film_manifest (
  id            BIGSERIAL PRIMARY KEY,
  project_id    BIGINT NOT NULL REFERENCES film_project(id) ON DELETE CASCADE,
  kind          TEXT NOT NULL,
  json          JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS film_qc (
  id            BIGSERIAL PRIMARY KEY,
  project_id    BIGINT NOT NULL REFERENCES film_project(id) ON DELETE CASCADE,
  report_json   JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Teacher traces → distill sets
CREATE TABLE IF NOT EXISTS teacher_trace (
  id            BIGSERIAL PRIMARY KEY,
  run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  trace_line    JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS distill_sft (
  id            BIGSERIAL PRIMARY KEY,
  run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  sample_json   JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS distill_dpo (
  id            BIGSERIAL PRIMARY KEY,
  run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  pair_json     JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS distill_toolpolicy (
  id            BIGSERIAL PRIMARY KEY,
  run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  policy_json   JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Ablation outputs
CREATE TABLE IF NOT EXISTS ablation_run (
  id            BIGSERIAL PRIMARY KEY,
  batch_uid     TEXT NOT NULL UNIQUE,
  config_json   JSONB NOT NULL,
  report_json   JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ablation_clean (
  id            BIGSERIAL PRIMARY KEY,
  ablation_id   BIGINT NOT NULL REFERENCES ablation_run(id) ON DELETE CASCADE,
  item_json     JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ablation_drop (
  id            BIGSERIAL PRIMARY KEY,
  ablation_id   BIGINT NOT NULL REFERENCES ablation_run(id) ON DELETE CASCADE,
  item_json     JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);


