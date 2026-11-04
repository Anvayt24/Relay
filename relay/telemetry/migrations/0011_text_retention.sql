-- Per-tenant retention of prompt and response text (PRD §8 Privacy: "per-tenant retention for
-- `full` logging"). Found in review: `retention_days` was declared and never enforced.
--
-- Only shadow pairs of `full`-logging tenants hold text. Each such row records when its text
-- expires, fixed when it is written (now + the tenant's retention_days), so the purge needs no
-- tenant config: the evidence worker clears every row whose time is up. A later change to a
-- tenant's retention applies to the pairs written after it.
ALTER TABLE eval_pairs ADD COLUMN text_expires_at timestamptz;
CREATE INDEX eval_pairs_text_expires_idx ON eval_pairs (text_expires_at)
    WHERE text_expires_at IS NOT NULL;
