-- Migration: Rename run_id to trace_id and change type to TEXT
-- This aligns tool_call with the trace_id type used in the run table

-- Drop the old index
DROP INDEX IF EXISTS tool_run_name_idx;

-- Drop the foreign key constraint first (PostgreSQL auto-generates constraint names)
DO $$
DECLARE
    constraint_name TEXT;
BEGIN
    -- Find the foreign key constraint on run_id column
    SELECT conname INTO constraint_name
    FROM pg_constraint c
    JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
    WHERE c.conrelid = 'tool_call'::regclass
      AND c.contype = 'f'
      AND a.attname = 'run_id';
    
    IF constraint_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE tool_call DROP CONSTRAINT %I', constraint_name);
    END IF;
END $$;

-- Add new trace_id column as TEXT
ALTER TABLE tool_call ADD COLUMN IF NOT EXISTS trace_id TEXT;

-- Populate trace_id from run table (only if run_id column exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tool_call' AND column_name = 'run_id') THEN
        UPDATE tool_call 
        SET trace_id = (SELECT trace_id FROM run WHERE run.id = tool_call.run_id)
        WHERE trace_id IS NULL AND run_id IS NOT NULL;
    END IF;
END $$;

-- Make trace_id NOT NULL (after population, only if there are no NULL values)
DO $$
BEGIN
    -- Only set NOT NULL if there are no NULL values
    IF NOT EXISTS (SELECT 1 FROM tool_call WHERE trace_id IS NULL) THEN
        ALTER TABLE tool_call ALTER COLUMN trace_id SET NOT NULL;
    END IF;
END $$;

-- Drop the old run_id column
ALTER TABLE tool_call DROP COLUMN IF EXISTS run_id;

-- Recreate the index with trace_id
CREATE INDEX IF NOT EXISTS tool_run_name_idx ON tool_call(trace_id, tool_name);

