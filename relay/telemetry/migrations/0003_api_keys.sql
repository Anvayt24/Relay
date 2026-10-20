-- Phase 5: API keys (PRD FR-3). Only HMAC-SHA256(pepper, key) is stored, never the key.
-- Tenants themselves live in the YAML config; this table maps a key to a tenant id.

CREATE TABLE api_keys (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- UNIQUE also creates the index every authenticated request looks up by.
    key_hash    bytea       NOT NULL UNIQUE,
    -- "rk_" + 8 chars in clear, so humans can tell keys apart ("revoke rk_3fA9xQ2u").
    key_prefix  text        NOT NULL,
    tenant      text        NOT NULL,
    -- Revoked, not deleted: the row stays as an audit trail of which keys existed.
    status      text        NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
    created_at  timestamptz NOT NULL DEFAULT now()
);
