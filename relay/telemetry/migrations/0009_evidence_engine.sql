-- Phase 12: the evidence engine (PRD §9.4, §10; ADRs 0019-0022).
--
-- eval_pairs gains what live shadow pairs need:
--   policy_version: the bundle that served the request (cards are per policy version);
--   *_text: prompt and answers, ONLY for tenants whose logging mode is `full`, so those pairs
--     can be human-labeled. NULL for metadata_only tenants (text_retained = false);
--   cost_nano_usd: list price of the shadow call plus the judge call (M8, the cost report).
ALTER TABLE eval_pairs ADD COLUMN policy_version text;
ALTER TABLE eval_pairs ADD COLUMN prompt_text    text;
ALTER TABLE eval_pairs ADD COLUMN candidate_text text;
ALTER TABLE eval_pairs ADD COLUMN reference_text text;
ALTER TABLE eval_pairs ADD COLUMN cost_nano_usd  bigint;

-- The worker reads one route's comparison in time order (the look sample is "the first
-- look * every judged pairs").
CREATE INDEX eval_pairs_route_idx ON eval_pairs (route, comparison, ts);

-- One row per (route, comparison, cohort): the latest card, overwritten by each worker run.
-- The headline fields are columns, for SQL and dashboards; `card` holds the whole §10.8 card.
CREATE TABLE route_cards (
    route              text        NOT NULL,
    comparison         text        NOT NULL,  -- cheap_vs_strong | fallback:a->b | band:lo-hi | canary:v
    cohort             text        NOT NULL,  -- a cohort key, "other", or "all"
    state              text        NOT NULL
        CHECK (state IN ('safe', 'unsafe', 'undecided', 'uncalibrated', 'stale')),
    estimate           double precision,
    ci_low             double precision,
    ci_high            double precision,
    floor              double precision NOT NULL,
    n_judge            integer     NOT NULL,
    n_human            integer     NOT NULL,
    look               integer     NOT NULL,
    recommended_action text        NOT NULL,
    policy_version     text,
    card               jsonb       NOT NULL,
    updated_at         timestamptz NOT NULL,
    PRIMARY KEY (route, comparison, cohort)
);
