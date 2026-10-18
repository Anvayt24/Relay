import pytest

from relay.config.models import ALIASES, MODELS
from relay.pipeline.context import RequestContext
from relay.pipeline.stage import Continue, Reject, ShortCircuit
from relay.pipeline.stages import ProviderStage, RouteStage
from relay.providers.mock import MockBehavior, MockProvider
from relay.routing.static import StaticRouter
from relay.schemas.errors import ErrorClass


async def test_route_stage_sets_decision(ctx: RequestContext) -> None:
    result = await RouteStage(StaticRouter(MODELS, ALIASES)).process(ctx)
    assert result == Continue()
    assert ctx.decision is not None
    assert ctx.decision.model == "mock/strong"


async def test_route_stage_rejects_unknown_model(ctx: RequestContext) -> None:
    ctx.request = ctx.request.model_copy(update={"model": "nope"})
    result = await RouteStage(StaticRouter(MODELS, ALIASES)).process(ctx)
    assert isinstance(result, Reject)
    assert result.error.status == 404


async def test_provider_stage_sends_upstream_id_and_returns_relay_name(
    ctx: RequestContext,
) -> None:
    await RouteStage(StaticRouter(MODELS, ALIASES)).process(ctx)
    result = await ProviderStage(
        MODELS, {"mock": MockProvider()}, first_token_timeout_s=1.0
    ).process(ctx)

    assert isinstance(result, ShortCircuit)
    # The mock echoes the model it was called with; the stage rewrites it back so the
    # client sees the Relay name, never the upstream ID.
    assert result.response.model == "mock/strong"
    [attempt] = ctx.attempts
    assert attempt.provider == "mock"
    assert attempt.model == "mock/strong"
    assert attempt.error_class is None


async def test_provider_stage_maps_provider_error_to_reject(ctx: RequestContext) -> None:
    await RouteStage(StaticRouter(MODELS, ALIASES)).process(ctx)
    provider = MockProvider(MockBehavior(error=ErrorClass.TIMEOUT))
    result = await ProviderStage(MODELS, {"mock": provider}, first_token_timeout_s=1.0).process(ctx)

    assert isinstance(result, Reject)
    assert result.error.status == 504
    assert ctx.attempts[0].error_class is ErrorClass.TIMEOUT


async def test_provider_stage_requires_a_decision(ctx: RequestContext) -> None:
    with pytest.raises(RuntimeError, match="RouteStage"):
        await ProviderStage(MODELS, {"mock": MockProvider()}, first_token_timeout_s=1.0).process(
            ctx
        )


def test_provider_stage_validates_config_at_construction() -> None:
    # A model whose provider has no adapter is a config bug: fail at startup, not on the
    # first request that happens to use it.
    with pytest.raises(ValueError, match="mock"):
        ProviderStage(MODELS, {}, first_token_timeout_s=1.0)
