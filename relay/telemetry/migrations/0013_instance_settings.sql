-- Facts about this installation that must stay true across restarts. For now one: a fingerprint
-- of the key pepper that API keys were hashed with (relay/auth/bootstrap.py). The pepper itself
-- is never stored, only an HMAC of a fixed string under it, so it cannot be recovered from here.
CREATE TABLE instance_settings (
    key        text        PRIMARY KEY,
    value      text        NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
