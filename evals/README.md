# evals/

| Folder | What |
|---|---|
| `router/` | Router quality vs %-strong against random / always-cheap / always-strong |
| `cache/` | Semantic-cache precision/recall vs threshold on a labeled tune/test split |
| `evidence/` | Detection delay, interval coverage, and judge-noise experiments |

Each eval writes a `metrics.json` that CI compares against the committed baseline.
