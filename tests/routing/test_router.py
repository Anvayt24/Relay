"""TrainedRouter decisions (ADR 0017): thresholds, bypasses, and every failure -> strong."""

import time
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from prometheus_client import REGISTRY

from relay.config.models import ModelSpec
from relay.pipeline.context import RequestContext
from relay.reliability.candidates import plan_candidates
from relay.reliability.edges import SeededEdgeStates
from relay.routing.router import TrainedRouter

from ..conftest import TEST_CONFIG  # type: ignore[import-not-found]
from .conftest import StubEmbedder, chat, length_artifact, routed_config

CHEAP, STRONG = "mock/cheap", "mock/strong"
EASY = "hi"  # ~0.97 under the length artifact
HARD = "x" * 400  # ~0.03


def make_ctx(req: Any, config: Any) -> RequestContext:
    return RequestContext(
        request_id="r",
        tenant=config.tenants["test"],
        request=req,
        route="default",
        priority="interactive",
        cohort=None,
        policy_version="static-v0",
        deadline=time.monotonic() + 60,
        stream=False,
    )


def router(
    config: Any = None,
    *,
    artifact: Any = "default",
    embedder: Any = "default",
    models: dict[str, ModelSpec] | None = None,
) -> TrainedRouter:
    config = config or routed_config()
    return TrainedRouter(
        config,
        models or dict(config.models),
        artifacts={"default": length_artifact() if artifact == "default" else artifact},
        embedder=StubEmbedder() if embedder == "default" else embedder,
    )


def fallback_count(cause: str) -> float:
    return REGISTRY.get_sample_value("relay_router_fallback_total", {"cause": cause}) or 0.0


async def decide(r: TrainedRouter, req: Any, config: Any = None) -> Any:
    config = config or routed_config()
    return await r.decide(req, make_ctx(req, config))


# ---------------------------------------------------------------- the decision rule
async def test_easy_prompts_go_cheap_and_hard_ones_strong() -> None:
    r = router()
    easy, hard = await decide(r, chat(EASY)), await decide(r, chat(HARD))
    assert (easy.model, easy.tier, easy.reason) == (CHEAP, "cheap", "router")
    assert (hard.model, hard.tier, hard.reason) == (STRONG, "strong", "router")
    assert easy.score is not None and hard.score is not None and easy.score > hard.score
    assert easy.policy_version == "t-v1"  # the router's version is the policy of record


async def test_threshold_is_inclusive() -> None:
    probe = await decide(router(), chat(EASY))
    assert probe.score is not None
    at = routed_config(threshold=probe.score)  # exactly the score: still cheap
    above = routed_config(threshold=min(1.0, probe.score + 1e-6))
    assert (await decide(router(at), chat(EASY), at)).model == CHEAP
    assert (await decide(router(above), chat(EASY), above)).model == STRONG


# ---------------------------------------------------------------- what the router never touches
@pytest.mark.parametrize(
    ("model", "expected_model", "expected_reason"),
    [
        ("relay/cheap", CHEAP, "alias"),
        ("relay/strong", STRONG, "alias"),
        ("mock/cheap", CHEAP, "explicit"),  # explicit models bypass the router entirely
        ("mock/strong", STRONG, "explicit"),
    ],
)
async def test_only_the_router_alias_is_routed(
    model: str, expected_model: str, expected_reason: str
) -> None:
    stub = StubEmbedder()
    d = await decide(router(embedder=stub), chat(HARD, model=model))
    assert (d.model, d.reason, d.score) == (expected_model, expected_reason, None)
    assert stub.calls == 0  # not even an embedding


async def test_routes_without_a_router_are_untouched() -> None:
    plain = TEST_CONFIG
    r = TrainedRouter(plain, dict(plain.models), artifacts={}, embedder=StubEmbedder())
    d = await r.decide(chat(EASY), make_ctx(chat(EASY), plain))
    assert (d.model, d.reason) == (STRONG, "alias")  # relay/auto is still the static alias


async def test_a_cheap_routed_request_keeps_its_fallbacks() -> None:
    config = routed_config()
    d = await decide(router(config), chat(EASY), config)
    # `router` is not `explicit`, so ADR 0011's no-fallback rule doesn't apply.
    ctx = make_ctx(chat(EASY), config)
    plan = plan_candidates(
        config=config,
        models=config.models,
        route="default",
        tenant=ctx.tenant,
        decision=d,
        edge_states=SeededEdgeStates(config),
    )
    assert plan[0].model == CHEAP


# ---------------------------------------------------------------- every failure routes to strong
@pytest.mark.parametrize(
    ("kwargs", "cause"),
    [
        ({"artifact": None}, "no_artifact"),
        ({"embedder": None}, "embedder_off"),
        ({"embedder": StubEmbedder(ready=False)}, "embedder_not_ready"),
        ({"embedder": StubEmbedder(fail=True)}, "embed_error"),
        ({"embedder": StubEmbedder(name="another-model")}, "embedding_model_mismatch"),
        ({"artifact": length_artifact(dimensions=8)}, "bad_dimension"),
    ],
)
async def test_failures_route_to_strong_never_error(kwargs: dict[str, Any], cause: str) -> None:
    before = fallback_count(cause)
    d = await decide(router(**kwargs), chat(EASY))  # an *easy* prompt: cheap would be wrong here
    assert (d.model, d.tier, d.score) == (STRONG, "strong", None)
    assert d.reason == f"router_fallback:{cause}"
    assert fallback_count(cause) == before + 1


async def test_a_slow_embedder_times_out_to_strong() -> None:
    config = routed_config(embed_timeout_s=0.05)
    r = router(config, embedder=StubEmbedder(delay_s=1.0))
    started = time.monotonic()
    d = await decide(r, chat(EASY), config)
    assert d.reason == "router_fallback:embed_timeout" and d.model == STRONG
    assert time.monotonic() - started < 0.5  # it did not wait for the slow embedder


async def test_an_unavailable_cheap_model_routes_to_strong() -> None:
    config = routed_config()
    models = {k: v for k, v in config.models.items() if k != CHEAP}  # its provider has no key
    d = await decide(router(config, models=models), chat(EASY), config)
    assert d.reason == "router_fallback:cheap_unavailable" and d.model == STRONG


async def test_a_mismatched_artifact_never_scores_with_the_wrong_embedder() -> None:
    stub = StubEmbedder(name="bge-small")  # artifact says hash-embedder
    d = await decide(router(embedder=stub), chat(EASY))
    assert d.reason == "router_fallback:embedding_model_mismatch"
    assert stub.calls == 0  # refused *before* spending an embedding


# ---------------------------------------------------------------- embeddings
async def test_the_embedding_is_computed_once_and_kept_on_the_context() -> None:
    stub = StubEmbedder()
    config = routed_config()
    ctx = make_ctx(chat(EASY), config)
    r = router(config, embedder=stub)
    await r.decide(ctx.request, ctx)
    assert stub.calls == 1 and ctx.embedding is not None


async def test_an_embedding_the_cache_already_computed_is_reused() -> None:
    stub = StubEmbedder()
    config = routed_config()
    ctx = make_ctx(chat(EASY), config)
    ctx.embedding = await StubEmbedder().embed("hi")  # what CacheStage would have stored
    d = await router(config, embedder=stub).decide(ctx.request, ctx)
    assert stub.calls == 0 and d.reason == "router"


async def test_multi_turn_requests_route_on_the_last_user_message() -> None:
    req = chat("placeholder")
    req = req.model_copy(
        update={
            "messages": [
                *chat(HARD).messages,
                *chat("ok").messages[:0],
                chat(EASY).messages[0].model_copy(update={"role": "user"}),
            ]
        }
    )
    # The hard prompt is earlier history; the latest question is easy.
    assert (await decide(router(), req)).model == CHEAP


# ---------------------------------------------------------------- properties
@settings(max_examples=60, deadline=None)
@given(text=st.text(min_size=1, max_size=600))
async def test_decisions_are_deterministic_and_scores_are_probabilities(text: str) -> None:
    r = router()
    a, b = await decide(r, chat(text)), await decide(r, chat(text))
    assert (a.model, a.score) == (b.model, b.score)
    assert a.score is not None and 0.0 <= a.score <= 1.0


@settings(max_examples=40, deadline=None)
@given(short=st.integers(1, 50), extra=st.integers(1, 500))
async def test_a_longer_prompt_never_scores_higher_under_the_length_artifact(
    short: int, extra: int
) -> None:
    r = router()
    s, long_ = await decide(r, chat("a" * short)), await decide(r, chat("a" * (short + extra)))
    assert s.score is not None and long_.score is not None and long_.score <= s.score
