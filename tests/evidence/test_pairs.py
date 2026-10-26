"""Building evaluation pairs. The call-budget guard is the important part: building real
pairs spends real API calls, so the plan is counted and capped before the first one."""

import argparse
import json
import random
from pathlib import Path

import pytest

from relay.evidence.judge import LLMJudge
from relay.evidence.pairs import build, make_pair, planned_calls, prompt_hash, read_prompts

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
    args = argparse.Namespace(prompts=str(prompts), limit=40, max_calls=100, both_orders=False)
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
