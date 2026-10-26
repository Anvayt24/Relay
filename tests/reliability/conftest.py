import time
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from pathlib import Path

import pytest

from relay.config.loader import load_config
from relay.config.schema import RelayConfig
from relay.config.tenants import Tenant
from relay.pipeline.context import RequestContext
from relay.routing.base import RouteDecision
from relay.schemas.chat import ChatChunk, ChatRequest, ChatResponse
from relay.simulator.engine import Simulator
from relay.simulator.plan import MockBehavior

FALLBACK_CONFIG: RelayConfig = load_config(
    str(Path(__file__).parents[1] / "config" / "fallback.yaml")
)


class ScriptedProvider:
    """An adapter whose n-th call follows the n-th behavior (then ``then`` forever).

    MockProvider applies one behavior to every call; recovery scenarios ("fails twice, then
    works") need a sequence. ``None`` means a normal, successful answer.
    """

    name = "scripted"

    def __init__(self, *script: MockBehavior | None, then: MockBehavior | None = None) -> None:
        self._script = list(script)
        self._then = then
        self.calls = 0

    def _next(self) -> Simulator:
        behavior = self._script[self.calls] if self.calls < len(self._script) else self._then
        self.calls += 1
        return Simulator.scripted(behavior)

    async def complete(self, req: ChatRequest) -> ChatResponse:
        return await self._next().open(req).complete()

    async def stream(self, req: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        sim = self._next()
        async with aclosing(sim.open(req).stream()) as inner:
            async for chunk in inner:
                yield chunk

    async def aclose(self) -> None:
        return None


def ok() -> ScriptedProvider:
    return ScriptedProvider()


def failing(error: str, **kw: object) -> ScriptedProvider:
    from relay.schemas.errors import ErrorClass

    return ScriptedProvider(then=MockBehavior(error=ErrorClass(error), **kw))  # type: ignore[arg-type]


def make_ctx(
    *,
    route: str = "safe",
    tenant: str = "strict",
    model: str = "p/model",
    reason: str = "alias",
    stream: bool = False,
    deadline_s: float = 60.0,
    clock: Callable[[], float] = time.monotonic,
    config: RelayConfig = FALLBACK_CONFIG,
    tenant_override: Tenant | None = None,
    content: str = "hi",
) -> RequestContext:
    request = ChatRequest.model_validate(
        {"model": "relay/auto", "messages": [{"role": "user", "content": content}]}
    )
    ctx = RequestContext(
        request_id="rel-test",
        tenant=tenant_override or config.tenants[tenant],
        request=request,
        route=route,
        priority="interactive",
        cohort=None,
        policy_version="test",
        deadline=clock() + deadline_s,
        stream=stream,
    )
    spec = config.models[model]
    ctx.decision = RouteDecision(
        tier=spec.tier, model=model, score=None, reason=reason, policy_version="test"
    )
    return ctx


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def config() -> RelayConfig:
    return FALLBACK_CONFIG
