# Judge calibration on public data

Calibrated on public data (2023-era models and author-built adversarial pairs), not on Relay's own traffic. MT-Bench votes are preferences, while the rubric asks whether one answer is materially worse.

Pairs: 840 (0 not judged). Gate (M7): kappa >= 0.6 and loss recall >= 0.7 on >= 50 pairs.

| judge | rubric | data | n | kappa [95% CI] | loss recall [95% CI] | accuracy | status |
|---|---|---|---|---|---|---|---|
| anthropic/sonnet | v2 | all | 416 | 0.612 [0.539, 0.688] | 0.764 [0.709, 0.817] | 0.805 | calibrated |
| anthropic/sonnet | v2 | mtbench | 296 | 0.518 [0.424, 0.613] | 0.737 [0.673, 0.801] | 0.760 | uncalibrated |
| anthropic/sonnet | v2 | llmbar | 120 | 0.825 [0.709, 0.915] | 0.857 [0.750, 0.947] | 0.917 | calibrated |
| anthropic/sonnet | v2 | llmbar-natural | 60 | 0.895 [0.759, 1.000] | 0.917 [0.783, 1.000] | 0.950 | calibrated |
| anthropic/sonnet | v2 | llmbar-adversarial | 60 | 0.756 [0.570, 0.899] | 0.800 [0.633, 0.950] | 0.883 | calibrated |
| gemini/flash-lite | v1 | all | 420 | 0.494 [0.405, 0.572] | 0.765 [0.706, 0.821] | 0.748 | uncalibrated |
| gemini/flash-lite | v1 | mtbench | 300 | 0.387 [0.282, 0.488] | 0.773 [0.709, 0.831] | 0.703 | uncalibrated |
| gemini/flash-lite | v1 | llmbar | 120 | 0.698 [0.559, 0.815] | 0.735 [0.610, 0.850] | 0.858 | calibrated |
| gemini/flash-lite | v1 | llmbar-natural | 60 | 0.786 [0.607, 0.931] | 0.792 [0.609, 0.952] | 0.900 | calibrated |
| gemini/flash-lite | v1 | llmbar-adversarial | 60 | 0.612 [0.387, 0.815] | 0.680 [0.481, 0.864] | 0.817 | uncalibrated |

Human-human agreement (MT-Bench pairs with two votes, binary label): n = 148, kappa = 0.539, accuracy = 0.777. A judge is not expected to beat this.

Position bias (anthropic/sonnet): {"candidate_as_A": 0.46919431279620855, "candidate_as_B": 0.47804878048780486}

Position bias (gemini/flash-lite): {"candidate_as_A": 0.5488372093023256, "candidate_as_B": 0.5121951219512195}
