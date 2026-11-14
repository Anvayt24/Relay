-- Phase 13: policy bundles, the pointer, and what each request saw (PRD FR-11, §9.4; ADRs
-- 0023, 0024).

-- One immutable row per bundle version. `content` is the bundle (router artifacts embedded),
-- `content_sha256` a hash of its routes: a replica refuses a row whose content doesn't match.
CREATE TABLE policy_bundles (
    version        text        PRIMARY KEY,           -- YYYY-MM-DD.N
    content        jsonb       NOT NULL,
    content_sha256 text        NOT NULL,
    created_at     timestamptz NOT NULL,
    created_by     text        NOT NULL,
    note           text
);

-- The single pointer every replica polls. `id` is always true: at most one row.
CREATE TABLE policy_pointer (
    id                boolean     PRIMARY KEY DEFAULT true CHECK (id),
    active_version    text        REFERENCES policy_bundles (version),
    previous_version  text        REFERENCES policy_bundles (version),
    candidate_version text        REFERENCES policy_bundles (version),
    candidate_mode    text        CHECK (candidate_mode IN ('shadow', 'canary')),
    canary_pct        integer     NOT NULL DEFAULT 0 CHECK (canary_pct BETWEEN 0 AND 100),
    updated_at        timestamptz NOT NULL
);

-- Every pointer change: who, what, why. Written in the same transaction as the change.
CREATE TABLE policy_events (
    id     bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts     timestamptz NOT NULL,
    action text        NOT NULL,   -- activate | shadow | canary | promote | rollback
    actor  text        NOT NULL,   -- an operator, or evidence-worker (automatic rollback)
    before jsonb       NOT NULL,
    after  jsonb       NOT NULL,
    reason text
);

-- What each request saw: its canary arm (control | canary, NULL when no canary ran), and in
-- shadow mode the candidate's decision, logged but never applied.
ALTER TABLE requests ADD COLUMN canary_arm            text;
ALTER TABLE requests ADD COLUMN shadow_policy_version text;
ALTER TABLE requests ADD COLUMN shadow_tier           text;
ALTER TABLE requests ADD COLUMN shadow_score          real;
