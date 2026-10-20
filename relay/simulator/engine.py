"""The simulator engine: one codebase behind both front doors (PRD §11.1).

    in-process:  Relay -> ProviderStage -> MockProvider -----+
                                                             |
    over HTTP:   any client -> server.py (OpenAI/Anthropic) -+--> Simulator.open() -> Call

``open()`` runs everything that decides a request's fate, synchronously and in order:
1. assign the request index (the next number, or one given explicitly by an experiment);
2. reject an unknown model (``model_not_found``, as a real provider would);
3. find the active incident, if any (wall clock);
4. charge the capacity buckets, refusing with a 429 when they are short (wall clock);
5. ask the planner for the ``Plan`` (pure).

The returned ``Call`` then *executes* the plan: it sleeps where the plan says and yields
chunks or raises ``ProviderError``. Gates first, then execution: a capacity 429 is instant, as
a real provider's is, while an injected 5xx arrives after the latency the plan drew.
"""

import asyncio
import itertools
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Self

from relay.schemas.chat import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    Choice,
    ChunkChoice,
    ChunkDelta,
    ResponseMessage,
    Usage,
)
from relay.schemas.errors import ErrorClass, ProviderError, ProviderSignal
from relay.simulator.capacity import CapacityTracker, Clock, LimitState, MonotonicClock
from relay.simulator.plan import (
    FinishReason,
    MockBehavior,
    Plan,
    Planner,
    SampledPlanner,
    ScriptedPlanner,
)
from relay.simulator.scenario import Incident, Scenario


class CapacityExceeded(ProviderError):
    """A 429 caused by the capacity buckets. It carries the limit state, so the HTTP layer
    can send the same ``x-ratelimit-*`` headers a real provider would."""

    def __init__(self, limits: LimitState) -> None:
        super().__init__(
            ErrorClass.RATE_LIMITED,
            f"rate limit reached; retry after {limits.retry_after_s:.3f}s",
            ProviderSignal(retry_after_s=limits.retry_after_s, limit_scope=limits.scope),
        )
        self.limits = limits


def prompt_tokens(req: ChatRequest) -> int:
    # Words, not real tokens: see ScriptedPlanner. Good enough for usage and TPM accounting.
    return sum(len(m.content.split()) for m in req.messages)


class Simulator:
    def __init__(
        self,
        planner: Planner,
        *,
        scenario: Scenario | None = None,
        seed: int = 0,
        clock: Clock | None = None,
    ) -> None:
        self.planner = planner
        self.scenario = scenario
        self.seed = seed
        self.clock = clock or MonotonicClock()
        self._capacity = (
            CapacityTracker(scenario.capacity, self.clock)
            if scenario is not None and scenario.capacity is not None
            else None
        )
        self._indexes = itertools.count()
        # Incident windows count from the first request, not from construction. A slow
        # startup (or a test that builds the simulator early) must not eat into the schedule.
        self._t0: float | None = None

    @classmethod
    def scripted(cls, behavior: MockBehavior | None = None) -> Self:
        return cls(ScriptedPlanner(behavior or MockBehavior()))

    @classmethod
    def from_scenario(
        cls, scenario: Scenario, *, seed: int = 0, clock: Clock | None = None
    ) -> Self:
        return cls(SampledPlanner(scenario, seed), scenario=scenario, seed=seed, clock=clock)

    def open(
        self,
        req: ChatRequest,
        *,
        api_key: str = "default",
        index: int | None = None,
        cohort: str = "default",
    ) -> "Call":
        if index is None:
            index = next(self._indexes)
        model = req.model
        if self.scenario is not None and model not in self.scenario.models:
            raise ProviderError(ErrorClass.MODEL_NOT_FOUND, f"The model `{model}` does not exist")

        incident = self._incident(model)
        limits = None
        if self._capacity is not None:
            limits = self._capacity.admit(api_key, model, prompt_tokens(req), req.max_tokens)
            if not limits.admitted:
                raise CapacityExceeded(limits)

        plan = self.planner.plan(req, model=model, index=index, cohort=cohort, incident=incident)
        return Call(req, plan, limits)

    def truth(self, model: str, cohort: str, index: int) -> bool:
        """The hidden correctness of (model, cohort, index): recomputed, never stored."""
        if self.scenario is None:
            raise LookupError("a scripted simulator has no ground truth")
        if model not in self.scenario.models:
            raise LookupError(f"unknown model {model!r}")
        return self.scenario.correct(self.seed, model, cohort, index)

    def _incident(self, model: str) -> Incident | None:
        if self.scenario is None or not self.scenario.incidents:
            return None
        now = self.clock.now()
        if self._t0 is None:
            self._t0 = now
        elapsed = now - self._t0
        # If windows overlap, the first one listed wins.
        return next((i for i in self.scenario.incidents if i.applies(model, elapsed)), None)


class Call:
    """One admitted request with its plan, ready to execute."""

    def __init__(self, req: ChatRequest, plan: Plan, limits: LimitState | None) -> None:
        self.req = req
        self.plan = plan
        self.limits = limits
        self.prompt_tokens = prompt_tokens(req)
        # The index is in the id, so a harness holding only the response can look up its
        # ground truth. The random suffix keeps ids unique across simulator restarts.
        self.id = f"chatcmpl-sim-{plan.index}-{uuid.uuid4().hex[:12]}"
        self.created = int(time.time())

    async def complete(self) -> ChatResponse:
        p = self.plan
        await _sleep(p.latency_s)
        if p.error is not None:
            await _sleep(p.hang_s)
            raise self._error(p.error)
        return ChatResponse(
            id=self.id,
            created=self.created,
            model=self.req.model,
            choices=[
                Choice(
                    message=ResponseMessage(content=" ".join(p.reply_words)),
                    finish_reason=p.finish_reason,
                )
            ],
            usage=self._usage(),
        )

    async def stream(self) -> AsyncGenerator[ChatChunk, None]:
        p = self.plan
        await _sleep(p.first_chunk_delay_s)
        if p.error is not None and p.fail_after_chunks is None:
            await _sleep(p.hang_s)
            raise self._error(p.error)

        yield self._chunk(ChunkDelta(role="assistant", content=""))
        for i, word in enumerate(p.reply_words):
            if p.fail_after_chunks == i:
                raise self._error(p.error or ErrorClass.STREAM_BROKEN)
            if p.stall_after_chunks == i:
                await asyncio.sleep(3600)  # "forever": only the reader's idle timeout ends this
            if i > 0:
                await _sleep(p.chunk_delay_s)
            yield self._chunk(ChunkDelta(content=word if i == 0 else f" {word}"))

        yield self._chunk(finish_reason=p.finish_reason)
        yield ChatChunk(
            id=self.id, created=self.created, model=self.req.model, choices=[], usage=self._usage()
        )

    def _chunk(
        self, delta: ChunkDelta | None = None, finish_reason: FinishReason | None = None
    ) -> ChatChunk:
        return ChatChunk(
            id=self.id,
            created=self.created,
            model=self.req.model,
            choices=[ChunkChoice(delta=delta or ChunkDelta(), finish_reason=finish_reason)],
        )

    def _usage(self) -> Usage:
        completion = len(self.plan.reply_words)
        return Usage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=completion,
            total_tokens=self.prompt_tokens + completion,
        )

    def _error(self, error_class: ErrorClass) -> ProviderError:
        signal = None
        if self.plan.retry_after_s is not None:
            signal = ProviderSignal(retry_after_s=self.plan.retry_after_s, limit_scope="model")
        return ProviderError(error_class, f"simulated {error_class}", signal)


async def _sleep(seconds: float) -> None:
    # asyncio.sleep, never time.sleep: a blocking sleep would freeze the event loop, and with
    # it every other request in this process.
    if seconds > 0:
        await asyncio.sleep(seconds)
