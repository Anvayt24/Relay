-- Phase 9: judged evaluation pairs and human labels (PRD §9.4, §10.3).
--
-- A pair is a candidate answer and a reference answer to one prompt, judged by an LLM. Pairs
-- from live traffic (Phase 12's shadow runner) carry request_id and route; calibration and
-- pilot pairs have neither. Prompt and response text is NOT stored here: only a hash, because
-- text is retained only for tenants whose logging mode is `full` (§9.4). The JSONL files hold
-- the text for pairs a human has to read.

CREATE TABLE eval_pairs (
    id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pair_id              text        NOT NULL UNIQUE,      -- stable id, e.g. pilot-0007
    request_id           text,                             -- live traffic only
    route                text,                             -- live traffic only
    prompt_hash          text        NOT NULL,
    cohort               text        NOT NULL,
    comparison           text        NOT NULL,             -- e.g. cheap_vs_strong
    candidate_model      text        NOT NULL,
    reference_model      text        NOT NULL,
    judge_label          text CHECK (judge_label IN ('win', 'tie', 'loss')),  -- NULL = abstained
    judge_model          text        NOT NULL,
    judge_prompt_version text        NOT NULL,
    position_swapped     boolean     NOT NULL,             -- candidate shown as response B
    text_retained        boolean     NOT NULL,
    ts                   timestamptz NOT NULL,
    source               text        NOT NULL              -- pilot | sim | public | shadow
);

CREATE INDEX eval_pairs_comparison_idx ON eval_pairs (comparison, cohort);

CREATE TABLE human_labels (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pair_id    text        NOT NULL REFERENCES eval_pairs (pair_id),
    label      text        NOT NULL CHECK (label IN ('win', 'tie', 'loss')),
    annotator  text        NOT NULL,
    is_relabel boolean     NOT NULL DEFAULT false,         -- the blind consistency re-check
    ts         timestamptz NOT NULL
);

CREATE INDEX human_labels_pair_idx ON human_labels (pair_id);
