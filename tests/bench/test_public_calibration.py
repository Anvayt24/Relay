"""Calibrating the judge on public human-labeled data (evals/judge/public): the conversion of
MT-Bench and LLMBar into Relay's pairs and labels, the judging run (pacing, a call cap,
resume, and a fallback judge that never mixes into the first judge's numbers), and the
per-judge report. No network and no real model: fake adapters only."""

import random
from collections import Counter
from typing import Any

import pytest

from evals.judge.build_pairs import PatientAdapter
from evals.judge.public import data, report, run
from relay.evidence.judge import LLMJudge
from relay.schemas.errors import ErrorClass, ProviderError

from ..evidence.test_rubric_and_judge import FakeAdapter


def vote(
    qid: int, a: str, b: str, winner: str, judge: str = "author_1", turn: int = 1
) -> dict[str, Any]:
    prompt = f"question {qid}"
    return {
        "question_id": qid,
        "model_a": a,
        "model_b": b,
        "winner": winner,
        "judge": judge,
        "turn": turn,
        "conversation_a": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{a} answers {qid}"},
        ],
        "conversation_b": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{b} answers {qid}"},
        ],
    }


# ---------------------------------------------------------------- MT-Bench conversion
def test_the_weaker_model_is_always_the_candidate_and_labels_follow_it() -> None:
    rows = [
        vote(1, "gpt-4", "alpaca-13b", "model_a"),  # the stronger won: the candidate lost
        vote(2, "alpaca-13b", "gpt-4", "model_a"),  # the weaker won, shown as model_a
        vote(3, "llama-13b", "claude-v1", "tie"),
    ]
    out = data.mtbench_set(rows, n=10, seed=0)
    by_q = {p.prompt_text: p for p in out.pairs}
    labels = {row["pair_id"]: row["label"] for row in out.labels}
    p1 = by_q["question 1"]
    assert (p1.candidate_model, p1.reference_model) == ("alpaca-13b", "gpt-4")
    assert p1.candidate_text == "alpaca-13b answers 1"
    assert p1.reference_text == "gpt-4 answers 1"
    assert labels[p1.pair_id] == "loss"
    assert labels[by_q["question 2"].pair_id] == "win"
    p3 = by_q["question 3"]
    assert p3.candidate_model == "llama-13b" and labels[p3.pair_id] == "tie"


def test_only_first_turn_judgments_are_used() -> None:
    # Relay's rubric judges one request and one answer; a second-turn vote is about a
    # conversation the judge would never see.
    rows = [vote(1, "gpt-4", "llama-13b", "model_a"), vote(2, "gpt-4", "llama-13b", "tie", turn=2)]
    out = data.mtbench_set(rows, n=10, seed=0)
    assert [p.prompt_text for p in out.pairs] == ["question 1"]


def test_a_pair_with_several_votes_keeps_one_and_sets_the_others_aside() -> None:
    # One human label per pair (calibrate reads one); the other annotators' votes are kept
    # for the human-human agreement the judge is compared against.
    rows = [
        vote(1, "gpt-4", "llama-13b", "model_a", judge="author_2"),
        vote(1, "llama-13b", "gpt-4", "tie", judge="author_1"),
    ]
    out = data.mtbench_set(rows, n=10, seed=0)
    [pair] = out.pairs
    assert [r["label"] for r in out.labels] == ["loss"]  # the first vote in the data
    assert [(r["pair_id"], r["label"], r["annotator"]) for r in out.extra_votes] == [
        (pair.pair_id, "tie", "mtbench:author_1")
    ]


def test_mtbench_sample_is_seeded_and_spread_over_questions() -> None:
    models = ["gpt-4", "claude-v1", "gpt-3.5-turbo", "vicuna-13b-v1.2", "alpaca-13b", "llama-13b"]
    rows = [
        vote(q, a, b, "model_a")
        for q in range(1, 11)
        for i, a in enumerate(models)
        for b in models[i + 1 :]
    ]  # 10 questions x 15 model pairs
    first = data.mtbench_set(rows, n=30, seed=7)
    again = data.mtbench_set(rows, n=30, seed=7)
    assert [p.pair_id for p in first.pairs] == [p.pair_id for p in again.pairs]
    per_question = Counter(p.prompt_text for p in first.pairs)
    assert set(per_question.values()) == {3}  # round-robin: every question, evenly
    assert len(data.mtbench_set(rows, n=30, seed=8).pairs) == 30


def test_an_unknown_model_is_refused_not_guessed() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        data.mtbench_set([vote(1, "gpt-4", "mystery-7b", "model_a")], n=1, seed=0)


# ---------------------------------------------------------------- LLMBar conversion
def test_llmbar_labels_are_from_the_candidates_side() -> None:
    items = [{"input": f"do {i}", "output_1": f"one {i}", "output_2": f"two {i}", "label": 1}
             for i in range(40)]  # fmt: skip
    pairs, labels = data.llmbar_set({"Natural": items}, per_subset={"Natural": 40}, seed=3)
    label_of = {row["pair_id"]: row["label"] for row in labels}
    assert len(pairs) == 40
    for p in pairs:
        candidate_is_better = p.candidate_text.startswith("one")  # output 1 is the better one
        assert label_of[p.pair_id] == ("win" if candidate_is_better else "loss")
        assert p.source == "llmbar-natural"
    # The candidate side is random (seeded), so both labels occur.
    assert set(label_of.values()) == {"win", "loss"}


def test_llmbar_adversarial_subsets_are_sampled_per_subset() -> None:
    def items(n: int) -> list[dict[str, Any]]:
        return [{"input": f"x{i}", "output_1": "a", "output_2": "b", "label": 2} for i in range(n)]

    subsets = {"Adversarial/Neighbor": items(30), "Adversarial/Manual": items(20)}
    pairs, _ = data.llmbar_set(subsets, per_subset={k: 15 for k in subsets}, seed=0)
    assert len(pairs) == 30
    assert {p.source for p in pairs} == {"llmbar-adversarial"}
    assert Counter(p.pair_id.rsplit("-", 1)[0] for p in pairs) == {
        "llmbar-adversarial-neighbor": 15,
        "llmbar-adversarial-manual": 15,
    }


def test_the_judging_order_mixes_the_sources() -> None:
    # Whatever prefix one judge finishes before a fallback takes over is then a fair sample.
    rows = [vote(q, "gpt-4", "llama-13b", "model_a") for q in range(1, 61)]
    mt = data.mtbench_set(rows, n=60, seed=0).pairs
    items = [{"input": f"x{i}", "output_1": "a", "output_2": "b", "label": 1} for i in range(60)]
    lb, _ = data.llmbar_set({"Natural": items}, per_subset={"Natural": 60}, seed=0)
    order = data.judging_order([*mt, *lb], seed=0)
    assert sorted(p.pair_id for p in order) == sorted(p.pair_id for p in [*mt, *lb])
    first_half = Counter(p.source for p in order[:60])
    assert 15 < first_half["mtbench"] < 45


# ---------------------------------------------------------------- the judging run
def unjudged(n: int) -> list[Any]:
    rows = [vote(q, "gpt-4", "llama-13b", "model_a") for q in range(1, n + 1)]
    return data.mtbench_set(rows, n=n, seed=0).pairs


def slot(name: str, adapter: FakeAdapter, pace_s: float = 0.0) -> run.JudgeSlot:
    counted = run.CountingAdapter(adapter)
    judge = LLMJudge(PatientAdapter(counted), model="m", name=name, rng=random.Random(0))
    return run.JudgeSlot(name=name, judge=judge, counter=counted, pace_s=pace_s)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


NEITHER = '{"verdict": "neither"}'


async def test_every_pair_is_judged_once_and_saved_as_it_goes() -> None:
    pairs = unjudged(3)
    saves: list[int] = []
    clock = Clock()
    summary = await run.judge_all(
        pairs,
        [slot("j1", FakeAdapter(NEITHER, NEITHER, NEITHER))],
        max_calls=10,
        save=lambda ps: saves.append(sum(bool(p.judge_model) for p in ps)),
        clock=clock,
        sleep=clock.sleep,
    )
    assert [p.judge_label for p in pairs] == ["tie", "tie", "tie"]
    assert {p.judge_model for p in pairs} == {"j1"}
    assert saves == [1, 2, 3]  # written after every pair: a crash loses at most one
    assert summary.judged == {"j1": 3} and summary.stopped is None


async def test_a_resumed_run_skips_pairs_already_judged() -> None:
    pairs = unjudged(3)
    clock = Clock()
    await run.judge_all(
        pairs,
        [slot("j1", FakeAdapter(NEITHER))],
        max_calls=1,
        save=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
    )
    adapter = FakeAdapter(NEITHER, NEITHER)
    await run.judge_all(
        pairs,
        [slot("j1", adapter)],
        max_calls=10,
        save=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
    )
    assert len(adapter.requests) == 2  # only the two that were left
    assert all(p.judge_model == "j1" for p in pairs)


async def test_calls_are_paced_under_the_rate_limit() -> None:
    pairs = unjudged(3)
    clock = Clock()
    await run.judge_all(
        pairs,
        [slot("j1", FakeAdapter(NEITHER, NEITHER, NEITHER), pace_s=4.0)],
        max_calls=10,
        save=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
    )
    assert clock.slept == [4.0, 4.0]  # nothing before the first call, then one gap per call


async def test_a_rate_limit_is_waited_out_not_counted_as_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evals.judge import build_pairs

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(build_pairs.asyncio, "sleep", no_sleep)
    pairs = unjudged(1)
    limited = ProviderError(ErrorClass.RATE_LIMITED, "slow down")
    clock = Clock()
    summary = await run.judge_all(
        pairs,
        [slot("j1", FakeAdapter(limited, limited, NEITHER))],
        max_calls=10,
        save=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
    )
    assert pairs[0].judge_label == "tie" and pairs[0].judge_model == "j1"
    assert summary.upstream_calls == 3  # the two 429s are real requests too


async def test_when_the_first_judges_quota_ends_the_fallback_judges_the_rest() -> None:
    pairs = unjudged(3)
    quota = ProviderError(ErrorClass.QUOTA_EXHAUSTED, "per day limit")
    first = slot("gemini", FakeAdapter(NEITHER, quota))
    second = slot("groq", FakeAdapter(NEITHER, NEITHER))
    clock = Clock()
    summary = await run.judge_all(
        pairs, [first, second], max_calls=10, save=lambda _: None, clock=clock, sleep=clock.sleep
    )
    # Each pair records its own judge: the two are calibrated separately, never pooled.
    assert [p.judge_model for p in pairs] == ["gemini", "groq", "groq"]
    assert summary.judged == {"gemini": 1, "groq": 2}
    assert summary.switches == [("gemini", "groq", 1)]


async def test_failed_calls_are_retried_never_saved_as_abstentions() -> None:
    # LLMJudge turns a failed call into label=None. Saved, it would look like the judge had
    # declined to answer, and the pair would never be judged again.
    pairs = unjudged(1)
    down = ProviderError(ErrorClass.AUTH, "bad key")  # not retried by the patient adapter
    clock = Clock()
    summary = await run.judge_all(
        pairs,
        [slot("j1", FakeAdapter(down, NEITHER))],
        max_calls=10,
        save=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
    )
    assert pairs[0].judge_label == "tie"
    assert summary.failed_calls == {"j1": 1}


async def test_repeated_failures_move_to_the_fallback_judge() -> None:
    pairs = unjudged(1)
    down = ProviderError(ErrorClass.AUTH, "bad key")
    first = slot("gemini", FakeAdapter(down, down, down))
    second = slot("groq", FakeAdapter(NEITHER))
    clock = Clock()
    summary = await run.judge_all(
        pairs,
        [first, second],
        max_calls=10,
        save=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
        failures_before_switch=3,
    )
    assert pairs[0].judge_model == "groq"
    assert summary.switches == [("gemini", "groq", 0)]


async def test_a_real_abstention_is_kept() -> None:
    # The judge answered, but not with a verdict: that is a measurement, kept as such.
    pairs = unjudged(1)
    clock = Clock()
    await run.judge_all(
        pairs,
        [slot("j1", FakeAdapter("I cannot decide."))],
        max_calls=10,
        save=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
    )
    assert pairs[0].judge_model == "j1" and pairs[0].judge_label is None


async def test_the_call_cap_stops_the_run_cleanly() -> None:
    pairs = unjudged(5)
    clock = Clock()
    summary = await run.judge_all(
        pairs,
        [slot("j1", FakeAdapter(*[NEITHER] * 5))],
        max_calls=2,
        save=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
    )
    assert sum(bool(p.judge_model) for p in pairs) == 2
    assert summary.stopped is not None and "cap" in summary.stopped


async def test_with_every_judge_exhausted_the_run_stops_and_keeps_its_work() -> None:
    pairs = unjudged(2)
    quota = ProviderError(ErrorClass.QUOTA_EXHAUSTED, "per day limit")
    clock = Clock()
    summary = await run.judge_all(
        pairs,
        [slot("j1", FakeAdapter(NEITHER, quota))],
        max_calls=10,
        save=lambda _: None,
        clock=clock,
        sleep=clock.sleep,
    )
    assert pairs[0].judge_model == "j1" and not pairs[1].judge_model
    assert summary.stopped is not None and "no judge left" in summary.stopped


# ---------------------------------------------------------------- the report
def test_each_judge_is_calibrated_only_on_its_own_pairs() -> None:
    from dataclasses import replace

    pairs = unjudged(120)
    labels = [{"pair_id": p.pair_id, "label": "loss", "annotator": "h", "is_relabel": False}
              for p in pairs]  # fmt: skip
    judged = [
        replace(p, judge_model="gemini" if i < 100 else "groq", judge_prompt_version="v1",
                judge_label="loss")
        for i, p in enumerate(pairs)
    ]  # fmt: skip
    out = report.build(judged, labels, extra_votes=[], n_boot=50)
    assert out["judges"]["gemini"]["all"]["n"] == 100
    assert out["judges"]["groq"]["all"]["n"] == 20
    assert out["judges"]["gemini"]["by_source"]["mtbench"]["n"] == 100


def test_human_human_agreement_is_measured_on_the_double_voted_pairs() -> None:
    labels = [{"pair_id": f"p{i}", "label": "loss" if i % 2 else "tie", "annotator": "a",
               "is_relabel": False} for i in range(60)]  # fmt: skip
    # The second annotator agrees on 50 of 60 pairs.
    extra = [{"pair_id": f"p{i}", "label": ("loss" if i % 2 else "tie") if i < 50 else
              ("tie" if i % 2 else "loss"), "annotator": "b", "is_relabel": False}
             for i in range(60)]  # fmt: skip
    agreement = report.human_agreement(labels, extra)
    assert agreement["n"] == 60
    assert agreement["accuracy"] == pytest.approx(50 / 60)


def test_the_download_waits_out_a_rate_limit_instead_of_crashing() -> None:
    import httpx

    replies = [httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(503),
               httpx.Response(200, json={"ok": True})]  # fmt: skip
    client = httpx.Client(transport=httpx.MockTransport(lambda _: replies.pop(0)))
    slept: list[float] = []
    assert data._get_json(client, "https://example.test/x", sleep=slept.append) == {"ok": True}
    assert slept == [3.0, 4.0]  # the server's Retry-After, then exponential backoff


# ---------------------------------------------------------------- cost control (paid judges)
def priced_slot(name: str, adapter: FakeAdapter, usd_per_call: float) -> run.JudgeSlot:
    s = slot(name, adapter)
    s.price = lambda usage: usd_per_call  # what one call cost, from its reported usage
    return s


async def test_the_dollar_cap_stops_a_paid_judge() -> None:
    pairs = unjudged(10)
    clock = Clock()
    summary = await run.judge_all(
        pairs, [priced_slot("sonnet", FakeAdapter(*[NEITHER] * 10), usd_per_call=0.4)],
        max_calls=100, max_usd=1.0, save=lambda _: None, clock=clock, sleep=clock.sleep,
    )  # fmt: skip
    # 0.4 + 0.4 + 0.4: the third call takes it past $1, and nothing more is started.
    assert sum(bool(p.judge_model) for p in pairs) == 3
    assert summary.spent_usd == pytest.approx(1.2)
    assert summary.stopped is not None and "spend cap" in summary.stopped


async def test_a_probe_judges_only_the_first_few_pairs() -> None:
    pairs = unjudged(5)
    clock = Clock()
    summary = await run.judge_all(
        pairs, [slot("j1", FakeAdapter(NEITHER, NEITHER))], max_calls=100, limit=2,
        save=lambda _: None, clock=clock, sleep=clock.sleep,
    )  # fmt: skip
    assert sum(bool(p.judge_model) for p in pairs) == 2
    assert summary.stopped is not None and "limit" in summary.stopped


def test_the_sonnet_judge_runs_without_thinking_or_temperature() -> None:
    # Thinking is on by default on Claude Sonnet 5.5 and billed as output; a verdict and a
    # one-line reason don't need it. And Sonnet 5.5 rejects a non-default temperature.
    from relay.schemas.chat import ChatRequest

    adapter = run.NoThinkingAnthropic(base_url="http://anthropic.test", api_key="k")
    req = ChatRequest.model_validate(
        {"model": "claude-sonnet-5-5", "messages": [{"role": "user", "content": "x"}],
         "max_tokens": 300}
    )  # fmt: skip
    body = adapter._body(req)
    assert body["thinking"] == {"type": "between_tools"}
    assert "temperature" not in body
