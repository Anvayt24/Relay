from pathlib import Path

import pytest

from relay.config.loader import load_config
from relay.config.models import ModelSpec
from relay.config.schema import RelayConfig, RouteSpec
from relay.pipeline.context import RequestContext
from relay.routing.static import STATIC_POLICY_VERSION, StaticRouter
from relay.schemas.errors import ErrorClass, RelayError

CONFIG = load_config(Path(__file__).parents[1] / "config" / "test.yaml")


def _router(config: RelayConfig = CONFIG) -> StaticRouter:
    return StaticRouter(config, config.models)


@pytest.mark.parametrize(
    ("alias", "model", "tier"),
    [
        ("relay/cheap", "mock/cheap", "cheap"),
        ("relay/strong", "mock/strong", "strong"),
        # No trained router yet: "auto" takes the quality-safe default (PRD FR-7).
        ("relay/auto", "mock/strong", "strong"),
    ],
)
async def test_aliases_resolve(ctx: RequestContext, alias: str, model: str, tier: str) -> None:
    decision = await _router().decide(ctx.request.model_copy(update={"model": alias}), ctx)
    assert decision.model == model
    assert decision.tier == tier
    assert decision.reason == "alias"
    assert decision.score is None
    assert decision.policy_version == STATIC_POLICY_VERSION


async def test_aliases_are_per_route(ctx: RequestContext) -> None:
    # The "cheap_only" route has no relay/strong: the same alias means different things (or
    # nothing) depending on the tenant's route (FR-3).
    ctx.route = "cheap_only"
    with pytest.raises(RelayError) as exc:
        await _router().decide(ctx.request.model_copy(update={"model": "relay/strong"}), ctx)
    assert exc.value.status == 404


async def test_explicit_model_passes_through(ctx: RequestContext) -> None:
    decision = await _router().decide(ctx.request.model_copy(update={"model": "mock/cheap"}), ctx)
    assert decision.model == "mock/cheap"
    assert decision.reason == "explicit"


async def test_unknown_model_is_404(ctx: RequestContext) -> None:
    with pytest.raises(RelayError) as exc:
        await _router().decide(ctx.request.model_copy(update={"model": "nope/model"}), ctx)
    assert exc.value.status == 404
    assert exc.value.error_class is ErrorClass.MODEL_NOT_FOUND
    assert exc.value.param == "model"


async def test_configured_but_unavailable_model_is_404(ctx: RequestContext) -> None:
    # In the config, but its provider has no adapter (no API key): same answer as unknown.
    router = StaticRouter(CONFIG, {"mock/cheap": CONFIG.models["mock/cheap"]})
    with pytest.raises(RelayError) as exc:
        await router.decide(ctx.request.model_copy(update={"model": "mock/strong"}), ctx)
    assert exc.value.status == 404


async def test_names_with_extra_slashes_resolve_by_full_name(ctx: RequestContext) -> None:
    # Upstream IDs contain slashes (openrouter/meta-llama/<model>:free). The registry is keyed
    # by the full Relay name, so nothing ever has to guess where the provider prefix ends.
    name = "openrouter/meta-llama/llama-3.3-70b:free"
    spec = ModelSpec(
        provider="mock",
        provider_model="meta-llama/llama-3.3-70b:free",
        tier="cheap",
        price_in=0,
        price_out=0,
    )
    config = CONFIG.model_copy(
        update={
            "models": {**CONFIG.models, name: spec},
            "routes": {**CONFIG.routes, "default": RouteSpec(aliases={})},
        }
    )
    decision = await _router(config).decide(ctx.request.model_copy(update={"model": name}), ctx)
    assert decision.model == name
