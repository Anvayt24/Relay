-- Runs once, when the Postgres data volume is first created.
-- Only extensions belong here. Tables are created by Relay's migrations (Phase 2),
-- so the schema is versioned in the repo instead of being baked into the container.
CREATE EXTENSION IF NOT EXISTS vector;
