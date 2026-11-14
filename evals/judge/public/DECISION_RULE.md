# Decision rule for choosing the judge

Written on 2026-11-07, **before** any rubric-v2 run. Its purpose is to stop the rule from being
chosen after the results are known. gemini/flash-lite (rubric v1) was already measured under
the same gate: it is listed here as the baseline, not as a candidate.

1. **Data.** All 420 pairs in `pairs.jsonl` and their labels in `labels.jsonl`, as written by
   `data.py`, pooled. No subset is selected afterwards. The per-source rows in the report are
   for reading, not for deciding.
2. **Gate.** M7, unchanged: κ ≥ 0.6 **and** loss recall ≥ 0.7, on at least 50 pairs, using the
   point estimates (the `status` that `calibrate` reports).
3. **One judge = one model + one rubric version.** Each is calibrated on its own run file only.
   If a fallback judge took over part of a run, its pairs are a separate judge and do not count
   toward the first judge's result.
4. **Eligibility.** A run must judge all 420 pairs. If it stops early (call or spend cap), it is
   reported and the decision waits for the rest. A judge that abstains on more than 5% of pairs
   is not eligible: a judge that often gives no verdict can't drive decisions.
5. **Candidates.** `gemini/flash` and `anthropic/sonnet` (Claude Sonnet 5.5, thinking off), both
   with rubric v2 (a short reason, then the verdict). Sonnet's spend is capped at **$3.00** list
   price.
6. **Choice.** Among the eligible judges that pass, the one with the lowest list cost per 1,000
   judgments, measured from the run's reported usage. On a tie, the higher loss recall wins.
   **If none passes, no judge is chosen**, and that is the result that gets reported.
7. **Reported next to the decision, not part of it:** results per source, the human-human
   agreement on MT-Bench (κ = 0.539, the ceiling there), position bias, and the abstention
   rate.

## Decision (2026-11-07)

**Judge: `anthropic/sonnet` (Claude Sonnet 5.5), rubric v2, thinking off.**

- It judged all 420 pairs (4 abstentions, 1%) and passes the gate pooled: κ = 0.612, loss
  recall = 0.764 (`calibration.json`). Its list cost was $1.84, about $4.40 per 1,000
  judgments.
- `gemini/flash` was dropped as a candidate by the user's decision, because it **could not be
  run**: the free tier's daily quota stopped it at 20 of 420 pairs. Its 20 verdicts were too
  few to compare, so the decision did not depend on them, and its partial run file is not
  kept in the repo.
- Caveat, reported with the result: κ's 95% interval is [0.539, 0.688]. The pass is on the
  point estimate, which is what rule 2 specifies, and it is narrow.
