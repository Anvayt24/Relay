"""One log row per request that reached the pipeline, with the right outcome (PRD FR-8).

Streaming outcomes (failed_midstream, cancelled) are covered in test_streaming_asgi.py, which
can break connections; here the cases go through a normal HTTP client.
"""

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from prometheus_client import REGISTRY

from relay.providers.mock import MockBehavior, MockProvider
from relay.schemas.errors import ErrorClass
from relay.telemetry.log_writer import MemorySink

CHAT = "/v1/chat/completions"
BODY = {"model": "relay/cheap", "messages": [{"role": "user", "content": "log me"}]}


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # ASGITransport does not run the lifespan, so run it explicitly. Leaving this block stops
    # the LogWriter, which flushes every queued row to the sink.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


async def test_success_row(make_app: Callable[..., FastAPI]) -> None:
    sink = MemorySink()
    app = make_app(MockProvider(), sink=sink)
    async with serving(app) as client:
        r = await client.post(
            CHAT, json=BODY, headers={"X-Request-ID": "log-1", "X-Relay-Cohort": "beta"}
        )
        assert r.status_code == 200

    [row] = sink.rows
    assert row.request_id == "log-1"
    assert row.status == "success"
    assert row.error_class is None
    assert (row.provider, row.model, row.tier) == ("mock", "mock/cheap", "cheap")
    assert (row.tokens_in, row.tokens_out) == (2, 3)
    assert row.cohort == "beta"
    assert row.stream is False
    assert row.ttft_ms is None
    assert row.latency_ms >= 0
    assert row.ts.tzinfo is not None  # timestamptz needs an aware datetime


async def test_provider_error_row(make_app: Callable[..., FastAPI]) -> None:
    sink = MemorySink()
    app = make_app(MockProvider(MockBehavior(error=ErrorClass.OVERLOADED)), sink=sink)
    async with serving(app) as client:
        assert (await client.post(CHAT, json=BODY)).status_code == 503

    [row] = sink.rows
    assert row.status == "error"
    assert row.error_class == "overloaded"
    assert row.provider == "mock"  # the attempt is recorded even though it failed
    assert row.tokens_in is None


async def test_unknown_model_row(make_app: Callable[..., FastAPI]) -> None:
    sink = MemorySink()
    app = make_app(MockProvider(), sink=sink)
    async with serving(app) as client:
        assert (await client.post(CHAT, json={**BODY, "model": "nope"})).status_code == 404

    [row] = sink.rows
    assert row.status == "error"
    assert row.error_class == "model_not_found"
    assert row.provider is None and row.model is None


async def test_streamed_row_has_ttft_and_usage(make_app: Callable[..., FastAPI]) -> None:
    sink = MemorySink()
    app = make_app(MockProvider(), sink=sink)
    async with serving(app) as client:
        r = await client.post(CHAT, json={**BODY, "stream": True})
        assert r.text.endswith("data: [DONE]\n\n")

    [row] = sink.rows
    assert row.stream is True
    assert row.status == "success"
    assert row.ttft_ms is not None
    # Usage was not requested by the client, but Relay still recorded it.
    assert row.tokens_out == 3


async def test_validation_errors_never_reach_the_pipeline(make_app: Callable[..., FastAPI]) -> None:
    sink = MemorySink()
    app = make_app(MockProvider(), sink=sink)
    async with serving(app) as client:
        assert (await client.post(CHAT, json={"model": "relay/cheap"})).status_code == 400

    # No RequestContext was ever built, so there is no row. This is documented, not a bug.
    assert list(sink.rows) == []


async def test_requests_succeed_when_postgres_is_unreachable(
    make_app: Callable[..., FastAPI],
) -> None:
    """The degradation-matrix row "Postgres down": the gateway keeps serving.

    Port 1 on localhost refuses connections. Startup must not try to connect (pool
    min_size=0), requests must not wait on the database, and shutdown must stay bounded.
    """
    settings: dict[str, Any] = {
        "database_url": "postgresql://relay:relay@127.0.0.1:1/relay",
        "log_flush_interval_s": 0.01,
        "log_shutdown_timeout_s": 0.5,
    }
    app = make_app(MockProvider(), **settings)
    before = _lost_rows()

    started = time.perf_counter()
    async with serving(app) as client:
        startup_s = time.perf_counter() - started
        for _ in range(5):
            r = await client.post(CHAT, json=BODY)
            assert r.status_code == 200
        ready = (await client.get("/readyz")).json()
        assert ready["status"] == "ready"

    assert startup_s < 0.5  # no connection attempt at startup
    assert time.perf_counter() - started < 5  # shutdown bounded, even with rows stuck
    # All 5 rows were lost, and every one was counted, either after the write retries ran out
    # (sink_failed) or when shutdown gave up waiting (shutdown).
    assert _lost_rows() == before + 5


def _lost_rows() -> float:
    total = 0.0
    for reason in ("sink_failed", "shutdown"):
        value = REGISTRY.get_sample_value("relay_log_dropped_total", {"reason": reason})
        total += value or 0.0
    return total
