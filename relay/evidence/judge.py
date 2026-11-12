"""Judges: turn (request, candidate, reference) into a win/tie/loss for the candidate.

- ``LLMJudge`` (the ``Judge`` protocol's implementation): asks a real model, through one of
  Relay's own adapters, with the rubric in ``rubric.py``. A/B order is randomized per call to
  cancel position bias (judges tend to favor whichever response comes first), and the order
  is recorded.
- The simulator's ``MockJudge`` (``relay/simulator/judge.py``) is *not* behind that protocol:
  it judges from the hidden ground truth (which model was secretly correct), not from text,
  with configurable noise. It is how the evals get a judge whose accuracy is *known*, so the
  calibration and the evidence engine can be checked against the truth.

The label is always from the **candidate's** point of view:
- candidate shown as A: ``A_worse`` = loss, ``B_worse`` = win, ``neither`` = tie;
- candidate shown as B: the reverse.
"""

import random
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from relay.evidence.rubric import JUDGE_PROMPT_VERSION, Verdict, build_messages, parse_judgment
from relay.providers.base import ProviderAdapter
from relay.schemas.chat import ChatRequest, Usage
from relay.schemas.errors import ProviderError

Label = Literal["win", "tie", "loss"]


@dataclass(frozen=True)
class JudgeResult:
    label: Label | None  # None = the judge abstained (unparseable, or the call failed)
    position_swapped: bool  # True = the candidate was shown as response B
    judge_model: str
    prompt_version: str
    raw: str = ""
    # What the judge call(s) used, as reported by the provider; None when nothing was reported
    # (or the call failed). The shadow budget is settled against it.
    usage: Usage | None = None
    # Why, in the judge's words (rubric v2); None for v1, a failed call, or no reason given.
    reason: str | None = None


class Judge(Protocol):
    async def judge(self, *, request: str, candidate: str, reference: str) -> JudgeResult: ...


def label_from_verdict(verdict: Verdict, *, candidate_is_a: bool) -> Label:
    if verdict == "neither":
        return "tie"
    candidate_worse = (verdict == "A_worse") == candidate_is_a
    return "loss" if candidate_worse else "win"


class LLMJudge:
    def __init__(
        self,
        adapter: ProviderAdapter,
        *,
        model: str,  # the upstream model id the adapter understands
        name: str,  # recorded on every result and route card (PRD §10.3)
        rng: random.Random | None = None,
        both_orders: bool = False,
        max_tokens: int = 16,
        prompt_version: str = JUDGE_PROMPT_VERSION,
        # 0 for the most repeatable verdicts. None sends no temperature at all: Claude Sonnet
        # 5.5 rejects any non-default value with a 400.
        temperature: float | None = 0.0,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._name = name
        self._rng = rng or random.Random()
        self._both_orders = both_orders
        self._max_tokens = max_tokens
        self._version = prompt_version
        self._temperature = temperature
        self.calls = 0  # real calls made: the pilot's call-budget guard reads this

    async def judge(self, *, request: str, candidate: str, reference: str) -> JudgeResult:
        swapped = self._rng.random() < 0.5
        first = await self._once(request, candidate, reference, swapped)
        if not self._both_orders or first.label is None:
            return first
        # Both orders (PRD §10.3): a verdict that flips with the order was position bias,
        # so a disagreement is a tie. Costs a second call per pair.
        second = await self._once(request, candidate, reference, not swapped)
        usage = _sum_usage(first.usage, second.usage)
        if second.label is None:
            return replace(second, usage=usage)
        if first.label != second.label:
            return JudgeResult("tie", swapped, self._name, self._version, first.raw, usage)
        return replace(first, usage=usage)

    async def _once(
        self, request: str, candidate: str, reference: str, swapped: bool
    ) -> JudgeResult:
        a, b = (reference, candidate) if swapped else (candidate, reference)
        messages = build_messages(request, a, b, version=self._version)
        body: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        self.calls += 1
        try:
            response = await self._adapter.complete(ChatRequest.model_validate(body))
        except ProviderError:
            return JudgeResult(None, swapped, self._name, self._version)
        text = response.choices[0].message.content or ""
        verdict, reason = parse_judgment(text)
        label = None if verdict is None else label_from_verdict(verdict, candidate_is_a=not swapped)
        return JudgeResult(
            label, swapped, self._name, self._version, text, response.usage, reason=reason
        )


def _sum_usage(a: Usage | None, b: Usage | None) -> Usage | None:
    if a is None or b is None:
        return a or b
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
    )
