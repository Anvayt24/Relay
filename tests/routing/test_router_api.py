"""The router through the real HTTP layer: headers, log rows, startup, and degradation."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI

from relay.api.app import create_app
from relay.config.loader import load_config
from relay.config.schema import RelayConfig, RouterSpec
from relay.providers.mock import MockProvider
from relay.routing.router import TrainedRouter
from relay.telemetry.log_writer import MemorySink

from ..conftest import TEST_CONFIG_PATH  # type: ignore[import-not-found]
from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]
from .conftest import StubEmbedder, length_artifact, routed_config

CHAT = "/v1/chat/completions"
EASY, HARD = "hi", "x" * 400


def post(content: str, model: str = "relay/auto", **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


def build(
    embedder: Any = "default", sink: MemorySink | None = None, artifact: Any = "default"
) -> tuple[FastAPI, MockProvider]:
    config = routed_config()
    provider = MockProvider()
    emb = StubEmbedder() if embedder == "default" else embedder
    art = length_artifact() if artifact == "default" else artifact
    router = TrainedRouter(config, dict(config.models), artifacts={"default": art}, embedder=emb)
    app = create_app(
        relay_settings(),
        providers={"mock": provider},
        config=config,
        router=router,
        embedder=emb,
        log_sink=sink,
    )
    return app, provider


async def test_easy_goes_cheap_hard_goes_strong_and_headers_say_so() -> None:
    app, _ = build()
    async with serving(app) as c:
        easy, hard = await c.post(CHAT, json=post(EASY)), await c.post(CHAT, json=post(HARD))
    assert easy.headers["x-relay-model"] == "mock/cheap"
    assert hard.headers["x-relay-model"] == "mock/strong"
    assert easy.json()["model"] == "mock/cheap"


async def test_the_decision_is_logged_with_score_reason_and_the_routers_version() -> None:
    sink = MemorySink()
    app, _ = build(sink=sink)
    async with serving(app) as c:
        await c.post(CHAT, json=post(EASY))
        await c.post(CHAT, json=post("hi", model="mock/strong"))  # explicit: bypasses it
        await c.post(CHAT, json=post("hi", model="relay/cheap"))
    routed, explicit, alias = list(sink.rows)
    assert (routed.router_reason, routed.model, routed.policy_version) == (
        "router",
        "mock/cheap",
        "t-v1",
    )
    assert routed.router_score is not None and 0.9 < routed.router_score <= 1.0
    # Everything else is logged too, with no score and the static policy.
    assert (explicit.router_reason, explicit.router_score) == ("explicit", None)
    assert (alias.router_reason, alias.router_score) == ("alias", None)
    assert explicit.policy_version == alias.policy_version == "static-v0"


async def test_explicit_models_and_other_aliases_bypass_the_router() -> None:
    stub = StubEmbedder()
    app, _ = build(embedder=stub)
    async with serving(app) as c:
        for model, expected in (("mock/cheap", "mock/cheap"), ("relay/strong", "mock/strong")):
            r = await c.post(CHAT, json=post(EASY, model=model))
            assert r.headers["x-relay-model"] == expected
    assert stub.calls == 0


async def test_a_broken_embedder_still_serves_every_request_from_strong() -> None:
    sink = MemorySink()
    app, provider = build(embedder=StubEmbedder(fail=True), sink=sink)
    async with serving(app) as c:
        r = await c.post(CHAT, json=post(EASY))
    assert r.status_code == 200 and r.headers["x-relay-model"] == "mock/strong"
    assert next(iter(sink.rows)).router_reason == "router_fallback:embed_error"
    assert provider.calls == 1


async def test_streams_are_routed_too() -> None:
    app, _ = build()
    async with serving(app) as c:
        r = await c.post(CHAT, json=post(EASY, stream=True))
    assert r.status_code == 200 and r.headers["x-relay-model"] == "mock/cheap"
    assert r.text.rstrip().endswith("[DONE]")


# ---------------------------------------------------------------- startup from a real file
def write_config(tmp_path: Path, artifact_name: str, threshold: float = 0.5) -> Path:
    """test.yaml plus a router on the default route, in a temp dir next to the artifact."""
    doc = yaml.safe_load(Path(TEST_CONFIG_PATH).read_text(encoding="utf-8"))
    doc["routes"]["default"]["router"] = {"artifact": artifact_name, "threshold": threshold}
    path = tmp_path / "relay.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


async def test_the_artifact_is_loaded_from_a_path_relative_to_the_config(tmp_path: Path) -> None:
    length_artifact().save(tmp_path / "router.json")
    cfg = write_config(tmp_path, "router.json")
    app = create_app(
        relay_settings(config_path=str(cfg), embedder="hash"), providers={"mock": MockProvider()}
    )
    async with serving(app) as c:
        easy, hard = await c.post(CHAT, json=post(EASY)), await c.post(CHAT, json=post(HARD))
    assert easy.headers["x-relay-model"] == "mock/cheap"
    assert hard.headers["x-relay-model"] == "mock/strong"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
async def test_a_missing_or_corrupt_artifact_does_not_stop_startup_and_routes_to_strong(
    tmp_path: Path, damage: str
) -> None:
    if damage == "corrupt":
        (tmp_path / "router.json").write_text(
            '{"format_version": 1, "weights": [', encoding="utf-8"
        )
    cfg = write_config(tmp_path, "router.json")
    sink = MemorySink()
    app = create_app(
        relay_settings(config_path=str(cfg), embedder="hash"),
        providers={"mock": MockProvider()},
        log_sink=sink,
    )
    async with serving(app) as c:
        r = await c.post(CHAT, json=post(EASY))
    assert r.status_code == 200 and r.headers["x-relay-model"] == "mock/strong"
    assert next(iter(sink.rows)).router_reason == "router_fallback:no_artifact"


async def test_embedder_off_means_strong_for_every_relay_auto_request(tmp_path: Path) -> None:
    length_artifact().save(tmp_path / "router.json")
    cfg = write_config(tmp_path, "router.json")
    app = create_app(
        relay_settings(config_path=str(cfg), embedder="off"), providers={"mock": MockProvider()}
    )
    async with serving(app) as c:
        r = await c.post(CHAT, json=post(EASY))
    assert r.headers["x-relay-model"] == "mock/strong"


async def test_a_route_without_a_router_still_serves_relay_auto_as_strong() -> None:
    app = create_app(relay_settings(), providers={"mock": MockProvider()})  # shipped test config
    async with serving(app) as c:
        r = await c.post(CHAT, json=post(EASY))
    assert r.headers["x-relay-model"] == "mock/strong"


# ---------------------------------------------------------------- config validation
def test_router_needs_the_cheap_and_strong_aliases() -> None:
    doc = yaml.safe_load(Path(TEST_CONFIG_PATH).read_text(encoding="utf-8"))
    # `cheap_only` has no relay/strong (and no relay/auto): a router there could not work.
    doc["routes"]["cheap_only"]["router"] = {"artifact": "a.json", "threshold": 0.5}
    with pytest.raises(ValueError, match=r"relay/strong|relay/auto"):
        RelayConfig.model_validate(doc)


@pytest.mark.parametrize(
    "bad",
    [
        {"artifact": "a.json", "threshold": 1.5},
        {"artifact": "a.json", "threshold": -0.1},
        {"artifact": "a.json", "threshold": 0.5, "embed_timeout_s": 0},
        {"artifact": "a.json", "threshold": 0.5, "unknown": 1},  # typos are errors
        {"threshold": 0.5},
    ],
)
def test_router_spec_rejects_bad_values(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RouterSpec(**bad)


def test_the_shipped_config_still_loads() -> None:
    assert load_config(None).routes["default"]
