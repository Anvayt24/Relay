# Evidence engine: real sanity check (NON-REPRODUCIBLE)

`python -m evals.evidence.real_check` on the real groq API. It shows the online path works end to end (gateway -> shadow -> judge -> pairs -> worker -> card); it is **not** a quality measurement: the judge is the same family as both models and has no human labels, so the card is `uncalibrated` by design and its estimate is judge-only.

- models: cheap `openai/gpt-oss-20b`, strong (and judge) `openai/gpt-oss-120b`
- requests: 30 of 30 public Dolly prompts sent; failed: 0
- real calls: 88 (budget 100); wall time 1182 s
- shadow pairs stored: 29; verdicts: win 5, tie 22, loss 2, abstained 0
- card state: **uncalibrated** (judge has not passed the M7 gate; estimate is judge-only)

| route | comparison | cohort | state | estimate | ci | floor | n judge | n human | look | action |
|---|---|---|---|---|---|---|---|---|---|---|
| check | cheap_vs_strong | all | uncalibrated | 0.931 | - | 0.9 | 29 | 0 | 2/3 | none |
