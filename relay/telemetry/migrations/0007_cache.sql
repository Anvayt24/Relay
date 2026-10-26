-- Phase 10: the semantic cache (PRD FR-6, §9.4; ADR 0015).
--
-- The cache stores embeddings and replies, never prompt text.
--
-- Enable pgvector here rather than relying on deploy/postgres/init.sql: that script only runs
-- when a data volume is first created, so a database that came from anywhere else (a CI service
-- container, a managed Postgres, an older volume) would fail this migration with
-- `type "vector" does not exist`. IF NOT EXISTS makes it a no-op where it is already enabled.
-- The server must ship the extension (the pgvector/pgvector images do), and creating it needs a
-- privileged role; on a managed database, an admin may need to enable it once.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE cache_entries (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- tenant + requested alias + system prompt + output-affecting parameters, hashed.
    -- A lookup only ever sees its own namespace (relay/cache/keys.py).
    namespace      text         NOT NULL,
    embedding      vector(384)  NOT NULL,
    reply          text         NOT NULL,   -- JSON: text, finish_reason, usage, served model
    -- Hash of the numbers and names in the prompt: the lexical guard compares these.
    guard          text         NOT NULL,
    served_model   text         NOT NULL,
    policy_version text         NOT NULL,
    created_at     timestamptz  NOT NULL DEFAULT now(),
    expires_at     timestamptz  NOT NULL
);

CREATE INDEX cache_entries_namespace_idx ON cache_entries (namespace);
CREATE INDEX cache_entries_expires_idx ON cache_entries (expires_at);
-- Approximate nearest neighbours on cosine distance. HNSW because it needs no training data
-- and builds incrementally as entries arrive.
CREATE INDEX cache_entries_embedding_idx ON cache_entries USING hnsw (embedding vector_cosine_ops);

ALTER TABLE requests ADD COLUMN cache_status text;  -- hit | miss | bypass (NULL = no cache stage)
