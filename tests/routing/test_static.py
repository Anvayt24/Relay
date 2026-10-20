import pytest

from relay.config.models import ALIASES, MODELS, ModelSpec
from relay.pipeline.context import RequestContext
from relay.routing.static import STATIC_POLICY_VERSION, StaticRouter
from relay.schemas.errors import ErrorClass, RelayError


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
    decision = await StaticRouter(MODELS, ALIASES).decide(
        ctx.request.model_copy(update={"model": alias}), ctx
    )
    assert decision.model == model
    assert decision.tier == tier
    assert decision.reason == "alias"
    assert decision.score is None
    assert decision.policy_version == STATIC_POLICY_VERSION


async def test_explicit_model_passes_through(ctx: RequestContext) -> None:
    decision = await StaticRouter(MODELS, ALIASES).decide(
        ctx.request.model_copy(update={"model": "mock/cheap"}), ctx
    )
    assert decision.model == "mock/cheap"
    assert decision.reason == "explicit"


async def test_unknown_model_is_404(ctx: RequestContext) -> None:
    with pytest.raises(RelayError) as exc:
        await StaticRouter(MODELS, ALIASES).decide(
            ctx.request.model_copy(update={"model": "nope/model"}), ctx
        )
    assert exc.value.status == 404
    assert exc.value.error_class is ErrorClass.MODEL_NOT_FOUND
    assert exc.value.param == "model"


async def test_names_with_extra_slashes_resolve_by_full_name(ctx: RequestContext) -> None:
    # Upstream IDs contain slashes (openrouter/meta-llama/<model>:free). The registry is keyed
    # by the full Relay name, so nothing ever has to guess where the provider prefix ends.
    name = "openrouter/meta-llama/llama-3.3-70b:free"
    models = {name: ModelSpec("openrouter", "meta-llama/llama-3.3-70b:free", "cheap")}
    decision = await StaticRouter(models, {}).decide(
        ctx.request.model_copy(update={"model": name}), ctx
    )
    assert decision.model == name


def test_alias_pointing_at_unknown_model_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="relay/broken"):
        StaticRouter(MODELS, {"relay/broken": "mock/missing"})
