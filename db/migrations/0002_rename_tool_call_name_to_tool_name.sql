-- Migration: Rename tool_call.name to tool_call.tool_name
-- This aligns the database schema with the application layer naming conventions

-- Drop the old index
DROP INDEX IF EXISTS tool_run_name_idx;

-- Rename the column (only if it exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tool_call' AND column_name = 'name') THEN
        ALTER TABLE tool_call RENAME COLUMN name TO tool_name;
    END IF;
END $$;

-- Recreate the index with the appropriate column (trace_id if exists, otherwise run_id)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tool_call' AND column_name = 'trace_id') THEN
        CREATE INDEX IF NOT EXISTS tool_run_name_idx ON tool_call(trace_id, tool_name);
    ELSIF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tool_call' AND column_name = 'run_id') THEN
        CREATE INDEX IF NOT EXISTS tool_run_name_idx ON tool_call(run_id, tool_name);
    END IF;
END $$;


