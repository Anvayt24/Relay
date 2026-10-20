"""Relay's ``mock`` provider: the simulator, used in-process (PRD §11.1: one codebase).

The adapter itself is thin. Everything about *what happens* to a request lives in
``relay.simulator``, and this class only adds what makes it an adapter:
- the ``ProviderAdapter`` shape;
- two counters that tests read.

There are two ways to build it:
- ``MockProvider(MockBehavior(...))``: a fixed script for every request. Deterministic and
  instant, so unit tests use it to reproduce one exact situation. This is the default.
- ``MockProvider(simulator=Simulator.from_scenario(...))``: seeded, sampled behavior (latency
  distributions, error mix, capacity, ground truth). ``RELAY_SIM_SCENARIO`` selects this.

Requests use cohort ``"default"``, because adapters only see the ``ChatRequest``, not
Relay's ``RequestContext``. Cohort-specific quality is reachable over HTTP (``X-Sim-Cohort``)
and from experiments that call the simulator directly.
"""

from collections.abc import AsyncGenerator
from contextlib import aclosing

from relay.schemas.chat import ChatChunk, ChatRequest, ChatResponse
from relay.simulator.engine import Simulator
from relay.simulator.plan import MockBehavior

# Re-exported: tests and callers have always imported MockBehavior from here.
__all__ = ["MockBehavior", "MockProvider"]


class MockProvider:
    name = "mock"

    def __init__(
        self, behavior: MockBehavior | None = None, *, simulator: Simulator | None = None
    ) -> None:
        if behavior is not None and simulator is not None:
            raise ValueError("pass a behavior or a simulator, not both")
        self.simulator = simulator or Simulator.scripted(behavior)
        # How many times complete()/stream() was called. Tests use it to prove nobody retried.
        self.calls = 0
        # Streams started but not yet closed. Tests use it to prove a disconnect closed the
        # upstream, instead of leaving it to the garbage collector.
        self.open_streams = 0

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        return await self.simulator.open(req).complete()

    async def stream(self, req: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        self.calls += 1
        self.open_streams += 1
        try:
            # open() runs inside the generator, so a gate failure (capacity 429, unknown
            # model) surfaces on the first chunk, like any other pre-stream error.
            #
            # aclosing(): when *this* generator is closed, `async for` stops but does NOT close
            # the inner generator; it would stay suspended until garbage collection. The same
            # trap applies to any async generator that wraps another one.
            async with aclosing(self.simulator.open(req).stream()) as inner:
                async for chunk in inner:
                    yield chunk
        finally:
            # Runs on normal end, on an exception, and on aclose() (GeneratorExit).
            self.open_streams -= 1

    async def aclose(self) -> None:
        return None  # in-process: no connections to release
