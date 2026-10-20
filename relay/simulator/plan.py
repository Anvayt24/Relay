"""The per-request decision: what will happen to this request, decided before anything runs.

A ``Plan`` holds every outcome of one request:
- how long until the first token, and the gap between tokens;
- the reply text and its finish_reason;
- which error, if any, and where in the stream it happens;
- whether the answer is secretly correct.

Planning is **pure**: no clock, no sleeping, no I/O. So:
- tests check the statistics of 20,000 plans in a fraction of a second;
- "same seed, same index ⇒ same plan" is a plain equality check;
- the engine stays a dumb executor that just sleeps and yields what the plan says.

There are two planners behind one protocol:
- ``ScriptedPlanner``: the same fixed ``MockBehavior`` for every request. Unit tests use it
  to reproduce one exact situation ("fail after 2 chunks"), and it is Phase 1-2's mock.
- ``SampledPlanner``: draws each request's plan from a ``Scenario``, with a seed. Chaos, load
  and evidence experiments use it.
"""

import math
import random
from dataclasses import dataclass
from typing import Literal, Protocol

from relay.schemas.chat import ChatRequest
from relay.schemas.errors import ErrorClass
from relay.simulator.scenario import Incident, Scenario, lognormal_params
from relay.simulator.seeding import rng_for, uniform_for

FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class MockBehavior:
    """A fixed script, applied to every request (the Phase 1-2 mock's knobs)."""

    latency_s: float = 0.0
    # Raise this class. Without fail_after_chunks it fails immediately, before any chunk.
    error: ErrorClass | None = None
    retry_after_s: float | None = None  # sent with the error, as a real 429 would
    reply_prefix: str = "mock: "

    # Streaming faults (PRD M1: slow first token, mid-stream disconnect, stall).
    first_chunk_delay_s: float = 0.0
    chunk_delay_s: float = 0.0
    # Fail after this many *content* chunks (0 = after the role chunk, before any text).
    # The class is `error`, or STREAM_BROKEN if unset.
    fail_after_chunks: int | None = None
    # Stop sending (without closing) after this many content chunks: a hung upstream.
    stall_after_chunks: int | None = None


@dataclass(frozen=True)
class Plan:
    index: int
    reply_words: tuple[str, ...]
    finish_reason: FinishReason
    # complete(): total time before the response. stream(): delay before the first chunk,
    # then between content chunks.
    latency_s: float = 0.0
    first_chunk_delay_s: float = 0.0
    chunk_delay_s: float = 0.0
    # Same semantics as MockBehavior: with fail_after_chunks None the error happens before
    # any chunk; otherwise mid-stream, before content chunk number fail_after_chunks.
    error: ErrorClass | None = None
    retry_after_s: float | None = None
    fail_after_chunks: int | None = None
    stall_after_chunks: int | None = None
    hang_s: float = 0.0  # a TIMEOUT waits this long before failing
    correct: bool | None = None  # ground truth; None for scripted plans


class Planner(Protocol):
    def plan(
        self,
        req: ChatRequest,
        *,
        model: str,
        index: int,
        cohort: str,
        incident: Incident | None,
    ) -> Plan: ...


class ScriptedPlanner:
    def __init__(self, behavior: MockBehavior) -> None:
        self.behavior = behavior

    def plan(
        self,
        req: ChatRequest,
        *,
        model: str,
        index: int,
        cohort: str,
        incident: Incident | None,
    ) -> Plan:
        b = self.behavior
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        # "Tokens" are whitespace-separated words. That is enough to exercise usage accounting
        # and max_tokens; real tokenizers arrive with admission estimates (Phase 6).
        words = f"{b.reply_prefix}{last_user}".split()
        words, finish_reason = _truncate(words, req.max_tokens)
        return Plan(
            index=index,
            reply_words=tuple(words),
            finish_reason=finish_reason,
            latency_s=b.latency_s,
            first_chunk_delay_s=b.first_chunk_delay_s,
            chunk_delay_s=b.chunk_delay_s,
            error=b.error,
            retry_after_s=b.retry_after_s,
            fail_after_chunks=b.fail_after_chunks,
            stall_after_chunks=b.stall_after_chunks,
        )


# Filler for sampled replies. The *content* of a simulated reply carries no meaning (its
# quality is the hidden `correct` flag); only its length and timing matter.
VOCABULARY = (
    "the", "model", "answers", "with", "a", "plausible", "sentence", "about", "this",
    "request", "and", "adds", "some", "detail", "so", "that", "the", "reply", "has", "a",
    "realistic", "length", "and", "shape", "for", "streaming", "tests",
)  # fmt: skip


class SampledPlanner:
    def __init__(self, scenario: Scenario, seed: int) -> None:
        self.scenario = scenario
        self.seed = seed

    def plan(
        self,
        req: ChatRequest,
        *,
        model: str,
        index: int,
        cohort: str,
        incident: Incident | None,
    ) -> Plan:
        profile = self.scenario.models[model]

        def rng(purpose: str) -> random.Random:
            # One substream per purpose: see seeding.py on common random numbers.
            return rng_for(self.seed, index, model, purpose)

        # --- timing
        lat = rng("latency")
        multiplier = incident.latency_multiplier if incident else 1.0
        latency = profile.latency
        ttft = 0.0
        if latency.ttft_median_s > 0:
            ttft = lat.lognormvariate(*lognormal_params(latency.ttft_median_s, latency.ttft_p95_s))
        jitter = latency.tokens_per_s_jitter
        rate = latency.tokens_per_s * lat.uniform(1 - jitter, 1 + jitter)
        chunk_delay = 0.0 if math.isinf(rate) else 1.0 / rate
        ttft, chunk_delay = ttft * multiplier, chunk_delay * multiplier

        # --- reply
        length = rng("length")
        median, p95 = profile.length.median_tokens, profile.length.p95_tokens
        n = max(1, round(length.lognormvariate(*lognormal_params(median, p95))))
        text = rng("text")
        words = [VOCABULARY[text.randrange(len(VOCABULARY))] for _ in range(n)]
        words, finish_reason = _truncate(words, req.max_tokens)

        # --- failure
        errors = incident.errors if incident and incident.errors else profile.errors
        error = errors.pick(uniform_for(self.seed, index, model, "error"))
        fail_after = None
        if error is ErrorClass.STREAM_BROKEN:
            # Somewhere inside the reply: before content chunk 0..n-1.
            u = uniform_for(self.seed, index, model, "break-at")
            fail_after = int(u * max(1, len(words)))

        if error is ErrorClass.TIMEOUT:
            # A timed-out request never produces a first token: the hang is its only wait.
            # (The latency draws above still happened, so other substreams are unaffected.)
            ttft = 0.0
        return Plan(
            index=index,
            reply_words=tuple(words),
            finish_reason=finish_reason,
            latency_s=0.0 if error is ErrorClass.TIMEOUT else ttft + chunk_delay * len(words),
            first_chunk_delay_s=ttft,
            chunk_delay_s=chunk_delay,
            error=error,
            retry_after_s=errors.retry_after_s if error is ErrorClass.RATE_LIMITED else None,
            fail_after_chunks=fail_after,
            hang_s=errors.hang_s if error is ErrorClass.TIMEOUT else 0.0,
            correct=self.scenario.correct(self.seed, model, cohort, index),
        )


def _truncate(words: list[str], max_tokens: int | None) -> tuple[list[str], FinishReason]:
    if max_tokens is not None and len(words) > max_tokens:
        return words[:max_tokens], "length"
    return words, "stop"
