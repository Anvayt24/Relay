"""Judges: turn (request, candidate, reference) into a win/tie/loss for the candidate.

Two implementations behind one protocol:
- ``LLMJudge``: asks a real model, through one of Relay's own adapters, with the rubric in
  ``rubric.py``. A/B order is randomized per call to cancel position bias (judges tend to
  favor whichever response comes first), and the order is recorded.
- the simulator's ``MockJudge`` (``relay/simulator/judge.py``): reads the hidden ground truth
  and adds configurable noise. It is how experiments get a judge whose accuracy is *known*,
  so the calibration and (later) the evidence engine can be validated against the truth.

The label is always from the **candidate's** point of view:
- candidate shown as A: ``A_worse`` = loss, ``B_worse`` = win, ``neither`` = tie;
- candidate shown as B: the reverse.
"""

import random
from dataclasses import dataclass
from typing import Literal, Protocol

from relay.evidence.rubric import JUDGE_PROMPT_VERSION, Verdict, build_messages, parse_verdict
from relay.providers.base import ProviderAdapter
from relay.schemas.chat import ChatRequest
from relay.schemas.errors import ProviderError

Label = Literal["win", "tie", "loss"]


@dataclass(frozen=True)
class JudgeResult:
    label: Label | None  # None = the judge abstained (unparseable, or the call failed)
    position_swapped: bool  # True = the candidate was shown as response B
    judge_model: str
    prompt_version: str
    raw: str = ""


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
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._name = name
        self._rng = rng or random.Random()
        self._both_orders = both_orders
        self._max_tokens = max_tokens
        self.calls = 0  # real calls made: the pilot's call-budget guard reads this

    async def judge(self, *, request: str, candidate: str, reference: str) -> JudgeResult:
        swapped = self._rng.random() < 0.5
        first = await self._once(request, candidate, reference, swapped)
        if not self._both_orders or first.label is None:
            return first
        # Both orders (PRD §10.3): a verdict that flips with the order was position bias,
        # so a disagreement is a tie. Costs a second call per pair.
        second = await self._once(request, candidate, reference, not swapped)
        if second.label is None:
            return second
        if first.label != second.label:
            return JudgeResult("tie", swapped, self._name, JUDGE_PROMPT_VERSION, first.raw)
        return first

    async def _once(
        self, request: str, candidate: str, reference: str, swapped: bool
    ) -> JudgeResult:
        a, b = (reference, candidate) if swapped else (candidate, reference)
        messages = build_messages(request, a, b)
        self.calls += 1
        try:
            response = await self._adapter.complete(
                ChatRequest.model_validate(
                    {
                        "model": self._model,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": self._max_tokens,
                    }
                )
            )
        except ProviderError:
            return JudgeResult(None, swapped, self._name, JUDGE_PROMPT_VERSION)
        text = response.choices[0].message.content or ""
        verdict = parse_verdict(text)
        label = None if verdict is None else label_from_verdict(verdict, candidate_is_a=not swapped)
        return JudgeResult(label, swapped, self._name, JUDGE_PROMPT_VERSION, text)
