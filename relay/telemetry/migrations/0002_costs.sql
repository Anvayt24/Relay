-- Phase 5: request cost (PRD FR-9). Money is integer nano-USD (1e-9 USD), never floating
-- point: sums over millions of rows must be exact. See relay/config/pricing.py.
-- Nullable: a request that never reached a provider has no usage, so no cost.

ALTER TABLE requests
    ADD COLUMN billed_cost_nano_usd           bigint,
    ADD COLUMN list_cost_nano_usd             bigint,
    -- "What always-strong would have cost": an estimate (same output length assumed).
    ADD COLUMN counterfactual_strong_nano_usd bigint,
    -- Which price table priced this row, so old rows stay explainable after prices change.
    ADD COLUMN price_table_version            text;
