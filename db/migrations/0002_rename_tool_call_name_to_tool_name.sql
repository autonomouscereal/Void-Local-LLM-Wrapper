-- Migration: Rename tool_call.name to tool_call.tool_name
-- This aligns the database schema with the application layer naming conventions

-- Drop the old index
DROP INDEX IF EXISTS tool_run_name_idx;

-- Rename the column
ALTER TABLE tool_call RENAME COLUMN name TO tool_name;

-- Recreate the index with the new column name
CREATE INDEX IF NOT EXISTS tool_run_name_idx ON tool_call(trace_id, tool_name);


