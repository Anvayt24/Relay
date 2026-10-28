"""Shadow sampling through the real gateway (ADR 0022): what gets sampled, privacy, budget,
and that the user's response never changes."""

import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI

from relay.admission.store import MemoryStore
from relay.api.app import create_app
from relay.config.schema import RelayConfig
from relay.evidence.cards import RouteCard
from relay.evidence.judge import JudgeResult
from relay.evidence.shadow import ShadowStage, band_comparison, jobs_for, request_text
from relay.evidence.store import MemoryEvidenceStore
from relay.pipeline.context import ProviderAttempt, RequestContext
from relay.pipeline.stage import Outcome
from relay.providers.mock import MockProvider
from relay.schemas.chat import ChatRequest
from relay.telemetry.log_writer import MemorySink

from ..conftest import TEST_CONFIG_PATH  # type: ignore[import-not-found]
from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

CHAT = "/v1/chat/completions"


def body(model: str = "relay/cheap", content: str = "what is 2+2?", **kw: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}], **kw}


class ScriptedJudge:
    def __init__(self, label: str | None = "tie") -> None:
        self.label = label
        self.calls: list[tuple[str, str, str]] = []

    async def judge(self, *, request: str, candidate: str, reference: str) -> JudgeResult:
        self.calls.append((request, candidate, reference))
        return JudgeResult(self.label, False, "judge/model", "v1")  # type: ignore[arg-type]


def evidence_config(
    *, evidence: dict[str, Any] | None = None, tenant: dict[str, Any] | None = None
) -> RelayConfig:
    with open(TEST_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["providers"]["judgep"] = {"kind": "mock"}
    raw["models"]["judge/model"] = {
        "provider": "judgep",
        "provider_model": "judge-v1",
        "tier": "strong",
        "price_in": 0.1,
        "price_out": 0.1,
    }
    route = raw["routes"]["default"]
    route["fallback_graph"] = {"mock/strong": [{"to": "mock/cheap"}]}
    route["evidence"] = {
        "judge": "judge/model",
        "sample_rate": 1.0,
        "sample_rate_decided": 1.0,
        **(evidence or {}),
    }
    raw["tenants"]["test"].update(
        {
            "evidence_sampling": "on",
            "allowed_providers": ["mock", "judgep"],
            "logging": "full",
            **(tenant or {}),
        }
    )
    return RelayConfig.model_validate(raw)


def build(
    config: RelayConfig,
    judge: ScriptedJudge | None = None,
    store: MemoryEvidenceStore | None = None,
    sink: MemorySink | None = None,
) -> tuple[FastAPI, MemoryEvidenceStore, ScriptedJudge, MockProvider]:
    store = store or MemoryEvidenceStore()
    judge = judge or ScriptedJudge()
    mock = MockProvider()
    app = create_app(
        relay_settings(),
        providers={"mock": mock, "judgep": MockProvider()},
        config=config,
        evidence_store=store,
        judges=lambda name, max_tokens: judge,
        log_sink=sink,
    )
    return app, store, judge, mock


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # Leaving the lifespan drains pending shadow tasks, so assertions after it see them all.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


async def test_a_cheap_answer_is_compared_against_the_strong_model() -> None:
    app, store, judge, mock = build(evidence_config())
    async with serving(app) as c:
        r = await c.post(CHAT, json=body())
    assert r.status_code == 200
    [pair] = store.pairs
    assert (pair.comparison, pair.candidate_model, pair.reference_model) == (
        "cheap_vs_strong",
        "mock/cheap",
        "mock/strong",
    )
    assert pair.judge_label == "tie" and pair.source == "shadow"
    assert pair.request_id == r.headers["x-request-id"]
    assert pair.cohort.startswith("lang=latin")
    assert pair.cost_nano_usd > 0  # the strong call and the judge, at list price
    assert mock.calls == 2  # the user's call and one shadow call
    [(request, candidate, reference)] = judge.calls
    assert reference  # the strong model's shadow answer
    assert candidate == r.json()["choices"][0]["message"]["content"]
    assert request == "what is 2+2?"


async def test_the_users_response_is_the_same_with_and_without_sampling() -> None:
    on_app, _, _, _ = build(evidence_config())
    off_cfg = evidence_config(tenant={"evidence_sampling": "off"})
    off_app, off_store, _, _ = build(off_cfg)
    async with serving(on_app) as c:
        on = [await c.post(CHAT, json=body(content=f"q{i}")) for i in range(5)]
        on_stream = await c.post(CHAT, json=body(stream=True))
    async with serving(off_app) as c:
        off = [await c.post(CHAT, json=body(content=f"q{i}")) for i in range(5)]
        off_stream = await c.post(CHAT, json=body(stream=True))
    for a, b in zip(on, off, strict=True):
        assert a.status_code == b.status_code == 200
        assert a.json()["choices"] == b.json()["choices"]
        assert a.json()["usage"] == b.json()["usage"]
    assert on_stream.text.count("data:") == off_stream.text.count("data:")
    assert off_store.pairs == []


async def test_a_stream_is_sampled_once_it_has_ended() -> None:
    app, store, judge, _ = build(evidence_config())
    async with serving(app) as c:
        r = await c.post(CHAT, json=body(stream=True))
        assert r.status_code == 200
    [pair] = store.pairs
    assert pair.comparison == "cheap_vs_strong"
    assert judge.calls[0][1]  # the streamed answer was collected and judged


async def test_a_healthy_primary_gets_evidence_for_its_fallback_edges() -> None:
    app, store, _, _ = build(evidence_config())
    async with serving(app) as c:
        await c.post(CHAT, json=body(model="relay/strong"))
    [pair] = store.pairs
    assert (pair.comparison, pair.candidate_model, pair.reference_model) == (
        "fallback:mock/strong->mock/cheap",
        "mock/cheap",
        "mock/strong",
    )


async def test_tenants_without_sampling_are_never_shadowed() -> None:
    app, store, judge, mock = build(evidence_config(tenant={"evidence_sampling": "off"}))
    async with serving(app) as c:
        await c.post(CHAT, json=body())
    assert store.pairs == [] and judge.calls == [] and mock.calls == 1


async def test_the_judge_must_be_a_provider_the_tenant_allows() -> None:
    # allowed_providers is a privacy rule: the prompt may not reach the judge's provider.
    cfg = evidence_config(tenant={"allowed_providers": ["mock"]})
    app, store, judge, mock = build(cfg)
    async with serving(app) as c:
        await c.post(CHAT, json=body())
    assert store.pairs == [] and judge.calls == [] and mock.calls == 1


async def test_the_daily_shadow_budget_caps_spend() -> None:
    app, store, _, mock = build(evidence_config(evidence={"shadow_budget_usd_per_day": 0}))
    async with serving(app) as c:
        await c.post(CHAT, json=body())
    assert store.pairs == [] and mock.calls == 1


async def test_metadata_only_tenants_store_no_text() -> None:
    app, store, _, _ = build(evidence_config(tenant={"logging": "metadata_only"}))
    async with serving(app) as c:
        await c.post(CHAT, json=body())
    [pair] = store.pairs
    assert not pair.text_retained
    assert pair.prompt_text is None and pair.candidate_text is None
    assert pair.prompt_hash  # the hash is kept, so pairs still join to their requests


async def test_full_logging_tenants_keep_text_for_human_labeling() -> None:
    app, store, _, _ = build(evidence_config())
    async with serving(app) as c:
        await c.post(CHAT, json=body())
    [pair] = store.pairs
    assert pair.text_retained and pair.prompt_text == "what is 2+2?"


async def test_an_abstaining_judge_still_stores_the_pair() -> None:
    app, store, _, _ = build(evidence_config(), judge=ScriptedJudge(label=None))
    async with serving(app) as c:
        await c.post(CHAT, json=body())
    [pair] = store.pairs
    assert pair.judge_label is None


async def test_cache_hits_are_not_shadowed() -> None:
    cfg = evidence_config(tenant={"cache": {"enabled": True}})
    app, store, _, _ = build(cfg)
    async with serving(app) as c:
        first = await c.post(CHAT, json=body(temperature=0))
        second = await c.post(CHAT, json=body(temperature=0))
    assert first.headers["x-relay-cache"] == "miss"
    assert second.headers["x-relay-cache"] == "hit"
    assert len(store.pairs) == 1


async def test_a_failed_request_is_not_shadowed() -> None:
    app, store, _, _ = build(evidence_config())
    async with serving(app) as c:
        r = await c.post(CHAT, json=body(model="no/such-model"))
    assert r.status_code == 404
    assert store.pairs == []


async def test_a_store_failure_is_swallowed() -> None:
    class Broken(MemoryEvidenceStore):
        async def write_pairs(self, rows: Any) -> None:
            raise RuntimeError("database down")

    app, _, _, _ = build(evidence_config(), store=Broken())
    async with serving(app) as c:
        r = await c.post(CHAT, json=body())
    assert r.status_code == 200


async def test_a_decided_card_lowers_the_sampling_rate() -> None:
    store = MemoryEvidenceStore()
    card = RouteCard(
        route="default", cohort="all", comparison="cheap_vs_strong", estimate=0.97,
        ci=(0.95, 0.99), floor=0.9, n_judge=2000, n_human=200, judge="judge/model@v1",
        judge_kappa=0.7, state="safe", recommended_action="none", look=4, max_looks=4,
        policy_version="p", updated_at=datetime.now(UTC),
    )  # fmt: skip
    await store.save_cards([card])
    cfg = evidence_config(evidence={"sample_rate_decided": 0.0})
    app, store, _, _ = build(cfg, store=store)
    async with serving(app) as c:
        await app.state.card_board.refresh()
        await c.post(CHAT, json=body())
    assert store.pairs == []


# ------------------------------------------------------------------ unit level
def _ctx(cfg: RelayConfig, **kw: Any) -> RequestContext:
    ctx = RequestContext(
        request_id="r1",
        tenant=cfg.tenants["test"],
        request=ChatRequest.model_validate(body(model="relay/auto")),
        route="default",
        priority="interactive",
        cohort=None,
        policy_version="p",
        deadline=1e12,
        stream=False,
    )
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


def test_band_evidence_covers_only_the_band_that_would_move() -> None:
    with open(TEST_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["routes"]["default"]["router"] = {"artifact": "x.json", "threshold": 0.8}
    raw["routes"]["default"]["evidence"] = {"judge": "mock/strong", "band_lo": 0.6}
    cfg = RelayConfig.model_validate(raw)
    route, spec = cfg.routes["default"], cfg.routes["default"].evidence
    assert spec is not None
    strong = [ProviderAttempt("mock", "mock/strong", 0.1, None)]

    def kinds(score: float | None, reason: str = "router") -> list[str]:
        ctx = _ctx(cfg, attempts=strong, router_reason=reason, router_score=score)
        return [j.comparison for j in jobs_for(ctx, route, spec)]

    assert kinds(0.7) == [band_comparison(0.6, 0.8)] == ["band:0.6-0.8"]
    assert kinds(0.6) == ["band:0.6-0.8"]  # lower edge included
    assert kinds(0.8) == []  # at the threshold it would have gone cheap already
    assert kinds(0.3) == []  # far below: a lower threshold wouldn't move it
    assert kinds(0.7, reason="explicit") == []  # only router decisions are in the band


def test_band_lo_must_sit_below_the_threshold() -> None:
    with open(TEST_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["routes"]["default"]["evidence"] = {"judge": "mock/strong", "band_lo": 0.6}
    with pytest.raises(ValueError, match="needs a router"):
        RelayConfig.model_validate(raw)
    raw["routes"]["default"]["router"] = {"artifact": "x.json", "threshold": 0.5}
    with pytest.raises(ValueError, match="below the router threshold"):
        RelayConfig.model_validate(raw)
    raw["routes"]["default"]["evidence"] = {"judge": "nope/model"}
    with pytest.raises(ValueError, match="unknown judge"):
        RelayConfig.model_validate(raw)


def test_answers_from_a_fallback_hop_are_not_evidence() -> None:
    cfg = evidence_config()
    spec = cfg.routes["default"].evidence
    assert spec is not None
    hop = [ProviderAttempt("mock", "mock/cheap", 0.1, None, hop=1)]
    assert jobs_for(_ctx(cfg, attempts=hop), cfg.routes["default"], spec) == []


def test_multi_turn_requests_are_shown_to_the_judge_as_a_transcript() -> None:
    req = ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "and?"},
            ],
        }
    )
    assert request_text(req) == "user: hi\nassistant: hello\nuser: and?"


async def test_overflowing_pending_tasks_are_dropped_not_queued() -> None:
    cfg = evidence_config()
    store = MemoryEvidenceStore()
    stage = ShadowStage(
        cfg,
        dict(cfg.models),
        judges=lambda name, max_tokens: ScriptedJudge(),
        store=store,
        budget=MemoryStore(),
        max_pending=0,
        rng=random.Random(0),
    )
    ctx = _ctx(
        cfg,
        attempts=[ProviderAttempt("mock", "mock/cheap", 0.1, None)],
        reply_parts=["4"],
    )
    await stage.finalize(ctx, Outcome("success"))
    await stage.drain()
    assert store.pairs == []


async def test_without_a_dispatcher_or_judge_nothing_is_called() -> None:
    cfg = evidence_config()
    store = MemoryEvidenceStore()
    stage = ShadowStage(
        cfg,
        dict(cfg.models),
        judges=lambda name, max_tokens: None,
        store=store,
        budget=MemoryStore(),
    )
    ctx = _ctx(cfg, attempts=[ProviderAttempt("mock", "mock/cheap", 0.1, None)], reply_parts=["4"])
    await stage.finalize(ctx, Outcome("success"))
    await stage.drain()
    assert store.pairs == []
