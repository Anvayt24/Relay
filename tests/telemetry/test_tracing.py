"""Traces (PRD FR-8; ADR 0025): one span tree per request, shaped like the pipeline, with the
attributes FR-8 lists, and nothing at all when tracing is off."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from relay.api.app import create_app
from relay.auth.keys import generate_key, hash_key
from relay.auth.service import Authenticator
from relay.auth.store import MemoryKeyStore
from relay.providers.mock import MockProvider
from relay.schemas.errors import ErrorClass, RelayError
from relay.simulator.plan import MockBehavior
from relay.telemetry import tracing

from ..conftest import TEST_CONFIG, TEST_PEPPER  # type: ignore[import-not-found]
from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]
from ..reliability.conftest import (  # type: ignore[import-not-found]
    FALLBACK_CONFIG,
    ScriptedProvider,
)

CHAT = "/v1/chat/completions"
BODY = {"model": "relay/strong", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.use_provider(provider)
    try:
        yield exporter
    finally:
        tracing.use_provider(None)


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


def by_name(spans: list[ReadableSpan]) -> dict[str, list[ReadableSpan]]:
    out: dict[str, list[ReadableSpan]] = {}
    for s in spans:
        out.setdefault(s.name, []).append(s)
    return out


def app_with(provider: Any = None, **settings: Any) -> FastAPI:
    return create_app(relay_settings(**settings), providers={"mock": provider or MockProvider()})


async def test_a_request_is_one_trace_shaped_like_the_pipeline(
    spans: InMemorySpanExporter,
) -> None:
    async with serving(app_with()) as c:
        r = await c.post(CHAT, json=BODY)
    assert r.status_code == 200
    finished = spans.get_finished_spans()
    names = by_name(list(finished))
    [root] = names["relay.request"]
    assert root.parent is None
    assert {s.context.trace_id for s in finished} == {root.context.trace_id}  # one trace
    stages = [s.name for s in finished if s.name.startswith("relay.stage.")]
    assert stages == [
        "relay.stage.log",
        "relay.stage.admission",
        "relay.stage.cache",
        "relay.stage.route",
        "relay.stage.reservation",
        "relay.stage.provider",
    ]
    assert all(s.parent.span_id == root.context.span_id for s in finished if s.name in stages)  # type: ignore[union-attr]
    [attempt] = names["relay.provider.attempt"]
    [provider_stage] = names["relay.stage.provider"]
    assert attempt.parent.span_id == provider_stage.context.span_id  # type: ignore[union-attr]
    assert attempt.attributes["relay.model"] == "mock/strong"  # type: ignore[index]
    # FR-8's attribute list, on the root span.
    attrs = root.attributes or {}
    assert attrs["relay.request_id"] == r.headers["x-request-id"]
    assert attrs["http.response.status_code"] == 200
    assert attrs["relay.cache.status"] == "bypass"
    assert attrs["relay.model"] == "mock/strong"
    assert attrs["relay.tier"] == "strong"
    assert attrs["relay.policy_version"] == "static-v0"
    assert attrs["relay.router.reason"] == "alias"
    assert attrs["relay.fallback_hops"] == 0
    assert attrs["relay.outcome"] == "success"


async def test_probes_and_scrapes_are_not_traced(spans: InMemorySpanExporter) -> None:
    # Found on the stack: healthchecks and Prometheus scrapes filled Tempo with noise.
    async with serving(app_with()) as c:
        for path in ("/healthz", "/readyz", "/metrics"):
            r = await c.get(path)
            assert r.status_code == 200 and "x-request-id" in r.headers
    assert spans.get_finished_spans() == ()


async def test_a_stream_is_traced_until_it_ends(spans: InMemorySpanExporter) -> None:
    async with serving(app_with()) as c:
        r = await c.post(CHAT, json={**BODY, "stream": True})
        assert "[DONE]" in r.text
    names = by_name(list(spans.get_finished_spans()))
    [root] = names["relay.request"]
    [provider_stage] = names["relay.stage.provider"]
    # The provider stage ends at the first chunk; the request span only when the stream does,
    # after finalize wrote the outcome onto it.
    assert root.end_time >= provider_stage.end_time  # type: ignore[operator]
    assert (root.attributes or {})["relay.outcome"] == "success"
    assert (root.attributes or {})["relay.stream"] is True


async def test_fallback_hops_are_separate_attempts_with_a_reason(
    spans: InMemorySpanExporter,
) -> None:
    provs = {
        "primary": ScriptedProvider(then=MockBehavior(error=ErrorClass.SERVER_ERROR)),
        "backup": ScriptedProvider(),
        "big": ScriptedProvider(),
        "capped": ScriptedProvider(),
    }
    app = create_app(
        relay_settings(auth_disabled_tenant="strict"), providers=provs, config=FALLBACK_CONFIG
    )
    async with serving(app) as c:
        r = await c.post(CHAT, json={**BODY, "model": "relay/auto"})
    assert r.status_code == 200
    attempts = by_name(list(spans.get_finished_spans()))["relay.provider.attempt"]
    seen = [
        (a.attributes["relay.model"], a.attributes["relay.hop"], a.attributes["relay.retry"])  # type: ignore[index]
        for a in attempts
    ]
    assert seen == [("p/model", 0, 0), ("p/model", 0, 1), ("b/model", 1, 0)]
    failed = attempts[0]
    assert failed.status.status_code is StatusCode.ERROR
    assert failed.attributes["relay.error_class"] == "server_error"  # type: ignore[index]
    assert attempts[2].attributes["relay.fallback.reason"] == "server_error"  # type: ignore[index]
    [root] = by_name(list(spans.get_finished_spans()))["relay.request"]
    assert (root.attributes or {})["relay.fallback_hops"] == 1


async def test_a_skipped_candidate_is_an_event(spans: InMemorySpanExporter) -> None:
    with tracing.tracer().start_as_current_span("parent"):
        tracing.skipped("b/model", "backup", 1, "breaker_open")
    [parent] = spans.get_finished_spans()
    [event] = parent.events
    assert event.name == "relay.candidate.skipped"
    assert event.attributes["relay.reason"] == "breaker_open"  # type: ignore[index]


async def test_a_callers_traceparent_is_honoured(spans: InMemorySpanExporter) -> None:
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    header = f"00-{trace_id}-00f067aa0ba902b7-01"
    async with serving(app_with()) as c:
        await c.post(CHAT, json=BODY, headers={"traceparent": header})
    [root] = by_name(list(spans.get_finished_spans()))["relay.request"]
    assert format(root.context.trace_id, "032x") == trace_id
    assert root.parent is not None and format(root.parent.span_id, "016x") == "00f067aa0ba902b7"


async def test_a_rejection_is_an_error_with_its_class(spans: InMemorySpanExporter) -> None:
    async with serving(app_with()) as c:
        r = await c.post(CHAT, json={**BODY, "model": "no/such-model"})
    assert r.status_code == 404
    names = by_name(list(spans.get_finished_spans()))
    [route] = names["relay.stage.route"]
    assert route.status.status_code is StatusCode.ERROR
    assert route.attributes["relay.error_class"] == "model_not_found"  # type: ignore[index]
    [root] = names["relay.request"]
    assert (root.attributes or {})["relay.error_class"] == "model_not_found"
    assert "relay.provider.attempt" not in names


async def test_the_auth_span_has_the_tenant_and_never_the_key(
    spans: InMemorySpanExporter,
) -> None:
    key = generate_key()
    auth = Authenticator(
        MemoryKeyStore({hash_key(key, TEST_PEPPER): "test"}), TEST_PEPPER, TEST_CONFIG
    )
    assert (await auth.authenticate(f"Bearer {key}")).id == "test"
    with pytest.raises(RelayError):
        await auth.authenticate("Bearer rk_wrong")
    ok, bad = spans.get_finished_spans()
    assert ok.name == bad.name == "relay.auth"
    assert (ok.attributes or {})["relay.tenant"] == "test"
    assert bad.status.status_code is StatusCode.ERROR
    assert (bad.attributes or {})["relay.error_class"] == "invalid_api_key"
    for span in (ok, bad):
        assert key not in str(span.attributes)


async def test_with_tracing_off_nothing_is_recorded() -> None:
    exporter = InMemorySpanExporter()
    assert not tracing.enabled()
    async with serving(app_with()) as c:
        r = await c.post(CHAT, json=BODY)
    assert r.status_code == 200 and exporter.get_finished_spans() == ()
    assert tracing.setup_tracing(relay_settings()) is None  # no endpoint: no provider


def test_an_endpoint_turns_tracing_on() -> None:
    provider = tracing.setup_tracing(
        relay_settings(otel_endpoint="http://127.0.0.1:1", otel_sample_ratio=0.5)
    )
    try:
        assert provider is not None and tracing.enabled()
        assert "0.5" in provider.sampler.get_description()
    finally:
        tracing.use_provider(None)
        if provider is not None:
            provider.shutdown()
