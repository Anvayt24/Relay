-- One row per request that reached the pipeline (PRD §9.4, FR-8). A subset of the PRD's
-- columns: later phases add cost, cache, fallback and evidence columns in new migrations.

CREATE TABLE requests (
    -- Surrogate key. request_id is NOT the primary key: clients can supply it (X-Request-ID)
    -- and may reuse it, and one duplicate would make the whole COPY batch fail.
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id  text        NOT NULL,
    ts          timestamptz NOT NULL,
    tenant      text        NOT NULL,
    route       text        NOT NULL,
    priority    text        NOT NULL,
    cohort      text,
    stream      boolean     NOT NULL,
    provider    text,
    model       text,
    tier        text,
    status      text        NOT NULL
        CHECK (status IN ('success', 'error', 'cancelled', 'failed_midstream')),
    error_class text,
    tokens_in   integer,
    tokens_out  integer,
    latency_ms  integer     NOT NULL,
    ttft_ms     integer
);

-- Most queries are "recent requests" (time range) or "what happened to this request".
CREATE INDEX requests_ts_idx ON requests (ts);
CREATE INDEX requests_request_id_idx ON requests (request_id);
