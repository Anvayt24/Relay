# evals/

| Folder | What | Phase |
|---|---|---|
| `router/` | Router quality vs %-strong against random / always-cheap / always-strong (PRD E-A) | 11 |
| `cache/` | Semantic-cache precision/recall vs threshold on a labeled tune/test split (PRD E-B) | 10 |
| `evidence/` | Detection delay, interval coverage, and judge-noise experiments (PRD §10.9) | 12 |

Each eval writes a `metrics.json` that CI compares against the committed baseline.
