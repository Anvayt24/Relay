-- Phase 7: what reliability did for each request (PRD §9.4).

-- How many fallback models were attempted (0 = the primary answered or nothing was tried),
-- whether an edge without evidence served it (X-Relay-Degraded), and which policy decided.
ALTER TABLE requests ADD COLUMN fallback_hops  integer NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN degraded       boolean NOT NULL DEFAULT false;
ALTER TABLE requests ADD COLUMN policy_version text;

-- One row per candidate the dispatcher considered, in order: every call (with its retry
-- number) and every skip (with the reason). Joined to requests on request_id. This is how
-- "why did this request end up on that model?" is answered after the fact.
CREATE TABLE attempts (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id     text        NOT NULL,
    ts             timestamptz NOT NULL,  -- when the request arrived (= requests.ts)
    hop            integer     NOT NULL,  -- 0 = primary, n = n-th fallback model attempted
    retry          integer     NOT NULL,  -- 0 = first try of this model, 1 = its retry
    provider       text        NOT NULL,
    model          text        NOT NULL,
    error_class    text,                  -- NULL with skipped_reason NULL = success
    skipped_reason text,                  -- set when the candidate was not called at all
    latency_ms     integer     NOT NULL
);

CREATE INDEX attempts_request_id_idx ON attempts (request_id);
CREATE INDEX attempts_ts_idx ON attempts (ts);
