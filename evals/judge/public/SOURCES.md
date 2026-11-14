# Sources and licences

The pairs and human labels in this folder are derived from two public datasets. Both licences
allow redistribution with attribution.

## MT-Bench human judgments

- Dataset: `lmsys/mt_bench_human_judgments` (Hugging Face), split `human`
- Licence: CC-BY-4.0
- Citation: Zheng et al., *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena*, 2023
  (arXiv:2306.05685)
- Used here: 300 first-turn pairs (seeded, spread over all 80 questions). Each pair keeps its
  first expert vote as the label (`labels.jsonl`); other annotators' votes on the same pair
  are in `extra_votes.jsonl`. Answers and votes are unchanged; only the format is Relay's.

## LLMBar

- Repository: `princeton-nlp/LLMBar` (GitHub), `Dataset/LLMBar/`
- Licence: MIT (Copyright (c) 2023 Princeton Natural Language Processing)
- Citation: Zeng et al., *Evaluating Large Language Models at Evaluating Instruction
  Following*, ICLR 2024 (arXiv:2310.07641)
- Used here: 120 instances: 60 from `Natural` and 15 from each adversarial subset
  (`Neighbor`, `GPTInst`, `GPTOut`, `Manual`). The candidate side is chosen by a seeded coin;
  the label is "win" if the candidate is the output LLMBar marks as better, else "loss".

## Reproducing

`python -m evals.judge.public.data` downloads both datasets and rewrites `pairs.jsonl`,
`labels.jsonl` and `extra_votes.jsonl` byte for byte (fixed seeds and timestamps). The judge's
verdicts (`run.py`) are real model calls and are not reproducible byte for byte; they are
saved in `pairs.jsonl` once made.
