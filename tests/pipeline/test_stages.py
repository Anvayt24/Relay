from pathlib import Path

import pytest

from relay.admission.store import MemoryStore
from relay.config.loader import load_config
from relay.pipeline.context import RequestContext
from relay.pipeline.stage import Continue, Reject, ShortCircuit
from relay.pipeline.stages import ProviderStage, RouteStage
from relay.providers.mock import MockBehavior, MockProvider
from relay.reliability.breaker import BreakerRegistry
from relay.reliability.dispatch import ReliableDispatcher
from relay.reliability.edges import SeededEdgeStates
from relay.reliability.retry_budget import RetryBudget
from relay.routing.static import StaticRouter
from relay.schemas.errors import ErrorClass

from ..conftest import build_dispatcher  # type: ignore[import-not-found]

CONFIG = load_config(Path(__file__).parents[1] / "config" / "test.yaml")


async def test_route_stage_sets_decision(ctx: RequestContext) -> None:
    result = await RouteStage(StaticRouter(CONFIG, CONFIG.models), CONFIG).process(ctx)
    assert result == Continue()
    assert ctx.decision is not None
    assert ctx.decision.model == "mock/strong"


async def test_route_stage_rejects_unknown_model(ctx: RequestContext) -> None:
    ctx.request = ctx.request.model_copy(update={"model": "nope"})
    result = await RouteStage(StaticRouter(CONFIG, CONFIG.models), CONFIG).process(ctx)
    assert isinstance(result, Reject)
    assert result.error.status == 404


async def test_provider_stage_sends_upstream_id_and_returns_relay_name(
    ctx: RequestContext,
) -> None:
    await RouteStage(StaticRouter(CONFIG, CONFIG.models), CONFIG).process(ctx)
    result = await ProviderStage(build_dispatcher(CONFIG, {"mock": MockProvider()})).process(ctx)

    assert isinstance(result, ShortCircuit)
    # The mock echoes the model it was called with; the stage rewrites it back so the
    # client sees the Relay name, never the upstream ID.
    assert result.response.model == "mock/strong"
    [attempt] = ctx.attempts
    assert attempt.provider == "mock"
    assert attempt.model == "mock/strong"
    assert attempt.error_class is None


async def test_provider_stage_maps_provider_error_to_reject(ctx: RequestContext) -> None:
    await RouteStage(StaticRouter(CONFIG, CONFIG.models), CONFIG).process(ctx)
    provider = MockProvider(MockBehavior(error=ErrorClass.TIMEOUT))
    result = await ProviderStage(build_dispatcher(CONFIG, {"mock": provider})).process(ctx)

    # No fallback is configured for this route, so the provider's own class reaches the
    # client. A timeout is not retried on the same model (Appendix B).
    assert isinstance(result, Reject)
    assert result.error.status == 504
    [attempt] = ctx.attempts
    assert attempt.error_class is ErrorClass.TIMEOUT


async def test_provider_stage_requires_a_decision(ctx: RequestContext) -> None:
    with pytest.raises(RuntimeError, match="RouteStage"):
        await ProviderStage(build_dispatcher(CONFIG, {"mock": MockProvider()})).process(ctx)


def test_dispatcher_validates_config_at_construction() -> None:
    # A model whose provider has no adapter is a config bug: fail at startup, not on the
    # first request that happens to use it.
    with pytest.raises(ValueError, match="mock"):
        ReliableDispatcher(
            CONFIG,
            CONFIG.models,
            {},
            store=MemoryStore(),
            pause=None,
            breakers=BreakerRegistry(CONFIG.reliability),
            retry_budget=RetryBudget(ratio=0.1, minimum=3, window_s=10),
            edge_states=SeededEdgeStates(CONFIG),
            first_token_ceiling_s=1.0,
        )
