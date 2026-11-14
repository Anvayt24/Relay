# Router evaluation: real world

**Real data, uncalibrated judge.** Labels come from an LLM judge that has not passed the M7 gate (no human labels yet), the set is small, so treat every interval as wide and nothing here may drive a decision.

Router `router-real-v1` (embedding `BAAI/bge-small-en-v1.5`). Test split: 27 prompts, 0 of them gold-labeled.

## Headline (M4: lower 95% bound of policy quality >= 0.95, share to cheap >= 40%)

- Threshold, chosen on **validation** only: **0.76** 
- On **test**, **3.7%** of requests go to the cheap model (target 40%: NOT met).
- Quality vs judge only: **1.0000**, one-sided 95% lower bound **1.0000** (floor 0.95: met).
- Judge-only estimate (biased by the judge's errors): 1.0000.
- List-price cost saved vs always-strong: 1.9% (illustrative: 100 prompt + 300 completion tokens per request, list prices from relay.yaml).

## Baselines (test)

| policy | share cheap | quality |
|---|---|---|
| always strong | 0% | 1.0000 |
| always cheap | 100% | 0.5556 (0.370-0.741) |
| oracle (sees the truth) | 55.6% | 1.0000 |
| **router** | 3.7% | 1.0000 |

AUC (score vs the cheap answer being win-or-tie): 0.506.

## Quality vs share sent strong (test)

| threshold | share strong | router quality (95% CI) | random at same share |
|---|---|---|---|
| 0.00 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.05 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.10 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.15 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.20 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.25 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.30 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.35 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.40 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.45 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.50 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.55 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.60 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.65 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.70 | 0.0% | 0.5556 (0.370-0.741) | 0.5556 |
| 0.75 | 14.8% | 0.5926 (0.407-0.778) | 0.6214 |
| 0.80 | 100.0% | 1.0000 (1.000-1.000) | 1.0000 |
| 0.85 | 100.0% | 1.0000 (1.000-1.000) | 1.0000 |
| 0.90 | 100.0% | 1.0000 (1.000-1.000) | 1.0000 |
| 0.95 | 100.0% | 1.0000 (1.000-1.000) | 1.0000 |
| 1.00 | 100.0% | 1.0000 (1.000-1.000) | 1.0000 |

## Where the router sends traffic (test, by family)

| family | prompts | router sends cheap | cheap answer is win-or-tie |
|---|---|---|---|
| lang=latin,len=medium,code=false,tools=false | 2 | 50% | 100% |
| lang=latin,len=short,code=false,tools=false | 25 | 0% | 52% |

## Calibration (test): is a score of 0.8 really 80%?

| mean score | observed win-or-tie | n |
|---|---|---|
| 0.746 | 0.667 | 3 |
| 0.750 | 1.000 | 3 |
| 0.751 | 0.500 | 2 |
| 0.751 | 0.333 | 3 |
| 0.752 | 0.333 | 3 |
| 0.752 | 0.000 | 3 |
| 0.753 | 0.333 | 3 |
| 0.755 | 0.500 | 2 |
| 0.756 | 0.667 | 3 |
| 0.760 | 1.000 | 3 |
