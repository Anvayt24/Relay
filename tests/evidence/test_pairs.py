"""Building evaluation pairs. The call-budget guard is the important part: building real
pairs spends real API calls, so the plan is counted and capped before the first one."""

import argparse
import json
import random
from pathlib import Path

import pytest

from relay.evidence.judge import LLMJudge
from relay.evidence.pairs import (
    build,
    make_pair,
    planned_calls,
    prompt_hash,
    read_jsonl,
    read_prompts,
)

from ..conftest import TEST_CONFIG_PATH  # type: ignore[import-not-found]
from .test_rubric_and_judge import FakeAdapter


def test_planned_calls_counts_candidate_reference_and_judge() -> None:
    assert planned_calls(30, both_orders=False) == 90  # the Phase 9 pilot
    assert planned_calls(30, both_orders=True) == 120


async def test_build_refuses_a_plan_over_the_call_cap(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(json.dumps({"prompt": f"question {i}"}) for i in range(40)), encoding="utf-8"
    )
    assert read_prompts(prompts)[0] == "question 0"
    args = argparse.Namespace(
        prompts=str(prompts),
        limit=40,
        max_calls=100,
        both_orders=False,
        out=str(tmp_path / "o.jsonl"),
        resume=False,
    )
    # 40 prompts need 120 calls, above the cap of 100: refused before any model is touched
    # (no config is even loaded).
    with pytest.raises(SystemExit, match="above --max-calls"):
        await build(args)


async def test_make_pair_calls_candidate_reference_and_judge_once_each() -> None:
    candidate = FakeAdapter("candidate answer")
    reference = FakeAdapter("reference answer")
    judge_adapter = FakeAdapter('{"verdict": "neither"}')
    judge = LLMJudge(judge_adapter, model="j", name="judge-x", rng=random.Random(0))
    pair = await make_pair(
        index=7,
        prompt="What is 2+2?",
        candidate=candidate,
        candidate_upstream="c-up",
        candidate_name="cheap",
        reference=reference,
        reference_upstream="r-up",
        reference_name="strong",
        judge=judge,
        judge_model="judge-x",
        max_tokens=64,
        source="pilot",
    )
    calls = (len(candidate.requests), len(reference.requests), len(judge_adapter.requests))
    assert calls == (1, 1, 1)
    assert candidate.requests[0].model == "c-up"
    assert candidate.requests[0].max_tokens == 64
    assert pair.id == "pilot-0007"
    assert pair.judge_label == "tie"
    assert pair.prompt_hash == prompt_hash("What is 2+2?")
    assert pair.cohort == "lang=latin,len=short,code=false,tools=false"


async def test_patient_adapter_waits_out_a_rate_limit_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from relay.evidence import pairs as pairs_module
    from relay.schemas.errors import ErrorClass, ProviderError, ProviderSignal

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(pairs_module.asyncio, "sleep", fake_sleep)
    limited = ProviderError(ErrorClass.RATE_LIMITED, "slow down", ProviderSignal(retry_after_s=0.6))
    adapter = pairs_module.PatientAdapter(FakeAdapter(limited, limited, "finally"))
    req = pairs_module.ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    response = await adapter.complete(req)
    assert response.choices[0].message.content == "finally"
    assert sleeps == [1.1, 1.1]  # the provider's own Retry-After (+0.5 s), twice
    assert adapter.rate_limit_waits == 2


async def test_patient_adapter_does_not_retry_other_errors() -> None:
    from relay.evidence import pairs as pairs_module
    from relay.schemas.errors import ErrorClass, ProviderError

    adapter = pairs_module.PatientAdapter(FakeAdapter(ProviderError(ErrorClass.AUTH, "bad key")))
    req = pairs_module.ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    with pytest.raises(ProviderError):
        await adapter.complete(req)


# ---------------------------------------------------------------- quotas and resuming (Phase 11)
async def test_a_daily_quota_stops_the_run_instead_of_waiting_or_retrying() -> None:
    from relay.evidence import pairs as pm
    from relay.schemas.errors import ErrorClass, ProviderError

    req = pm.ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    quota = ProviderError(ErrorClass.QUOTA_EXHAUSTED, "free-models-per-day")
    adapter = pm.PatientAdapter(FakeAdapter(quota))
    with pytest.raises(pm.QuotaReached, match="free-models-per-day"):
        await adapter.complete(req)
    assert adapter.quota_hit


async def test_a_retry_after_too_long_to_sit_through_is_treated_as_a_quota() -> None:
    from relay.evidence import pairs as pm
    from relay.schemas.errors import ErrorClass, ProviderError, ProviderSignal

    req = pm.ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    hour = ProviderError(ErrorClass.RATE_LIMITED, "slow", ProviderSignal(retry_after_s=3600))
    adapter = pm.PatientAdapter(FakeAdapter(hour))
    with pytest.raises(pm.QuotaReached):
        await adapter.complete(req)


async def test_a_judge_quota_keeps_the_paid_for_answers_as_an_abstention() -> None:
    from relay.evidence import pairs as pm
    from relay.schemas.errors import ErrorClass, ProviderError

    judge_adapter = pm.PatientAdapter(
        FakeAdapter(ProviderError(ErrorClass.QUOTA_EXHAUSTED, "daily"))
    )
    judge = LLMJudge(judge_adapter, model="j", name="judge-x", rng=random.Random(0))
    pair = await make_pair(
        index=0,
        prompt="q",
        candidate=FakeAdapter("c"),
        candidate_upstream="c",
        candidate_name="cheap",
        reference=FakeAdapter("r"),
        reference_upstream="r",
        reference_name="strong",
        judge=judge,
        judge_model="judge-x",
        max_tokens=10,
        source="real",
    )
    # Both answers are kept (they cost calls); only the verdict is missing, and `repair` will
    # fill it in later for a single call.
    assert (pair.candidate_response, pair.reference_response, pair.judge_label) == ("c", "r", None)
    assert judge_adapter.quota_hit


async def test_resume_skips_prompts_that_are_already_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from relay.evidence import pairs as pm

    prompts = tmp_path / "p.jsonl"
    prompts.write_text(
        "\n".join(json.dumps({"prompt": f"question {i}"}) for i in range(4)), encoding="utf-8"
    )
    out = tmp_path / "pairs.jsonl"
    adapters: dict[str, FakeAdapter] = {}
    # First run: only 2 of the 4 prompts.
    monkeypatch.setattr(pm, "build_adapters", lambda *a, **k: adapters)
    base = dict(
        prompts=str(prompts), candidate="mock/cheap", reference="mock/strong", judge="mock/strong",
        max_tokens=10, judge_max_tokens=10, max_calls=100, both_orders=False, source="real",
        seed=0, config=TEST_CONFIG_PATH, out=str(out),
    )  # fmt: skip

    # Every role (candidate, reference, judge) shares one adapter object here, so its replies
    # are consumed in call order: candidate, reference, judge, for each prompt in turn.
    Same = FakeAdapter

    shared = Same("c0", "r0", '{"verdict": "neither"}', "c1", "r1", '{"verdict": "neither"}')
    adapters["mock"] = shared
    await pm.build(argparse.Namespace(**base, limit=2, resume=False))
    assert len(read_jsonl(out)) == 2
    # Second run with --resume and the full list: only the 2 remaining prompts are generated.
    shared2 = Same("c2", "r2", '{"verdict": "neither"}', "c3", "r3", '{"verdict": "neither"}')
    adapters["mock"] = shared2
    await pm.build(argparse.Namespace(**base, limit=4, resume=True))
    done = read_jsonl(out)
    assert [p.prompt for p in done] == [f"question {i}" for i in range(4)]
    assert len(shared2.requests) == 6  # 3 calls x 2 new prompts, nothing re-paid
    assert [p.id for p in done] == [f"real-{i:04d}" for i in range(4)]  # stable ids


async def test_without_a_retry_after_the_wait_backs_off_exponentially_up_to_a_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from relay.evidence import pairs as pm
    from relay.schemas.errors import ErrorClass, ProviderError

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(pm.asyncio, "sleep", fake_sleep)
    busy = ProviderError(ErrorClass.RATE_LIMITED, "temporarily rate-limited upstream")
    adapter = pm.PatientAdapter(FakeAdapter(*[busy] * 6, "ok"))
    req = pm.ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    assert (await adapter.complete(req)).choices[0].message.content == "ok"
    assert sleeps == [2.5, 4.5, 8.5, 16.5, 30.5, 30.5]  # +0.5 s margin each; capped at 30


async def test_transient_errors_are_retried_a_few_times_then_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from relay.evidence import pairs as pm
    from relay.schemas.errors import ErrorClass, ProviderError

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(pm.asyncio, "sleep", no_sleep)
    req = pm.ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    blip = ProviderError(ErrorClass.SERVER_ERROR, "upstream connection failed: ReadError")
    # Two dropped connections, then success: the run survives.
    ok = pm.PatientAdapter(FakeAdapter(blip, blip, "fine"))
    assert (await ok.complete(req)).choices[0].message.content == "fine"
    # A persistent failure still ends the run after the cap (3 retries = 4 attempts).
    bad = pm.PatientAdapter(FakeAdapter(*[blip] * 4))
    with pytest.raises(ProviderError):
        await bad.complete(req)
    # Errors that retrying can't fix are not retried at all.
    auth = pm.PatientAdapter(FakeAdapter(ProviderError(ErrorClass.AUTH, "bad key")))
    with pytest.raises(ProviderError):
        await auth.complete(req)
