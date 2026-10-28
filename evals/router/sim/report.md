# Router evaluation: sim world

**Simulated world.** Difficulty was built into the prompts (ADR 0018), so this measures the *machinery* (training, thresholding, bootstrap, PPI, the CI gate), not real-world savings.

Router `router-sim-v1` (embedding `BAAI/bge-small-en-v1.5`). Test split: 631 prompts, 150 of them gold-labeled.

## Headline (M4: lower 95% bound of policy quality >= 0.95, share to cheap >= 40%)

- Threshold, chosen on **validation** only: **0.82** 
- On **test**, **37.2%** of requests go to the cheap model (target 40%: NOT met).
- Quality vs ground truth: **0.9857**, one-sided 95% lower bound **0.9778** (floor 0.95: met).
- Judge-only estimate (biased by the judge's errors): 0.9699.
- **PPI-corrected** estimate: 0.9799, one-sided 95% lower bound 0.9630; the 150 gold labels are worth about 1.2x as many human labels (lambda 0.51).
- List-price cost saved vs always-strong: 33.5% (illustrative: 100 prompt + 300 completion tokens per request, the shipped mock prices).

## Baselines (test)

| policy | share cheap | quality |
|---|---|---|
| always strong | 0% | 1.0000 |
| always cheap | 100% | 0.7655 (0.734-0.797) |
| oracle (sees the truth) | 76.5% | 1.0000 |
| **router** | 37.2% | 0.9857 |

AUC (score vs the cheap answer being win-or-tie): 0.813.

## Quality vs share sent strong (test)

| threshold | share strong | router quality (95% CI) | random at same share |
|---|---|---|---|
| 0.00 | 0.0% | 0.7655 (0.734-0.797) | 0.7655 |
| 0.05 | 0.0% | 0.7655 (0.734-0.797) | 0.7655 |
| 0.10 | 0.0% | 0.7655 (0.734-0.797) | 0.7655 |
| 0.15 | 0.0% | 0.7655 (0.734-0.797) | 0.7655 |
| 0.20 | 0.0% | 0.7655 (0.734-0.797) | 0.7655 |
| 0.25 | 0.0% | 0.7655 (0.734-0.797) | 0.7655 |
| 0.30 | 0.0% | 0.7655 (0.734-0.797) | 0.7655 |
| 0.35 | 0.2% | 0.7670 (0.735-0.800) | 0.7658 |
| 0.40 | 1.9% | 0.7750 (0.743-0.807) | 0.7699 |
| 0.45 | 5.2% | 0.7956 (0.767-0.827) | 0.7777 |
| 0.50 | 7.9% | 0.8146 (0.786-0.845) | 0.7840 |
| 0.55 | 11.7% | 0.8399 (0.813-0.867) | 0.7930 |
| 0.60 | 16.3% | 0.8621 (0.835-0.891) | 0.8037 |
| 0.65 | 23.8% | 0.8875 (0.862-0.913) | 0.8212 |
| 0.70 | 34.2% | 0.9239 (0.903-0.945) | 0.8457 |
| 0.75 | 47.5% | 0.9620 (0.946-0.975) | 0.8770 |
| 0.80 | 58.3% | 0.9794 (0.967-0.989) | 0.9022 |
| 0.85 | 67.8% | 0.9921 (0.984-0.998) | 0.9245 |
| 0.90 | 76.7% | 0.9984 (0.995-1.000) | 0.9454 |
| 0.95 | 94.1% | 1.0000 (1.000-1.000) | 0.9862 |
| 1.00 | 100.0% | 1.0000 (1.000-1.000) | 1.0000 |

## Where the router sends traffic (test, by family)

| family | prompts | router sends cheap | cheap answer is win-or-tie |
|---|---|---|---|
| advice | 34 | 0% | 68% |
| arithmetic | 165 | 5% | 73% |
| code | 63 | 0% | 62% |
| conversion | 60 | 38% | 88% |
| definition | 40 | 95% | 100% |
| reasoning | 104 | 0% | 42% |
| translation | 88 | 100% | 98% |
| trivia | 77 | 100% | 100% |

## Calibration (test): is a score of 0.8 really 80%?

| mean score | observed win-or-tie | n |
|---|---|---|
| 0.445 | 0.375 | 64 |
| 0.584 | 0.531 | 64 |
| 0.659 | 0.594 | 64 |
| 0.703 | 0.734 | 64 |
| 0.742 | 0.734 | 64 |
| 0.784 | 0.859 | 64 |
| 0.833 | 0.859 | 64 |
| 0.892 | 0.969 | 64 |
| 0.926 | 1.000 | 64 |
| 0.954 | 0.984 | 64 |
