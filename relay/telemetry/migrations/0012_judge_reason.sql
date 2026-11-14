-- The judge's reason for its verdict (rubric v2: a short reason, then the verdict), so a human
-- reading a card can see why a pair was judged a loss. It can quote the request, so it is
-- text like prompt_text: kept only for `full`-logging tenants and cleared by the retention
-- purge with the other texts (relay/evidence/store.py).
ALTER TABLE eval_pairs ADD COLUMN judge_reason text;
