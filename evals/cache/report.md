# Semantic cache evaluation

2946 labeled pairs (1498 tune / 1448 test), embedding model `BAAI/bge-small-en-v1.5`. Labels are **constructed** (same fact in a different wording = acceptable; one fact changed or unrelated = not), not human-judged. The hit rate reflects how the set was built (about 43% of pairs are true paraphrases), not production traffic.

Threshold chosen on **tune** as the smallest with precision >= 99%; everything below is **test**, which that choice never saw.

| lexical guard | threshold | test precision | test hit rate (95% CI) | recall |
|---|---|---|---|---|
| off | 0.9450 | 99.22% (lower 95%: 98.00%) | 26.6% (24.3-29.0) | 61.5% |
| on | 0.8625 | 99.35% (lower 95%: 98.53%) | 42.7% (40.1-45.1) | 98.9% |

The production configuration is the guard **on**. Wrong hits on test, by kind: guard off {'near_miss': 3}, guard on {'near_miss': 4}.
