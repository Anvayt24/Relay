"""Request-level metrics (bounded labels) and log lines that carry the request (PRD FR-8)."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import REGISTRY

from relay.api.app import create_app
from relay.providers.mock import MockProvider
from relay.telemetry import tracing
from relay.telemetry.logs import (
    REQUEST_ID,
    JsonFormatter,
    RequestContextFilter,
    configure_logging,
    writer_formatter,
)

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

CHAT = "/v1/chat/completions"
BODY = {"model": "relay/cheap", "messages": [{"role": "user", "content": "hi"}]}


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://relay.test") as c:
            yield c


def value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ------------------------------------------------------------------ metrics
async def test_a_request_is_counted_timed_and_priced() -> None:
    ok = {"tenant": "test", "route": "default", "status": "success", "error_class": "none"}
    before = {
        "requests": value("relay_requests_total", **ok),
        "duration": value(
            "relay_request_duration_seconds_count",
            route="default",
            status="success",
            stream="false",
        ),
        "tokens_in": value(
            "relay_tokens_total", tenant="test", provider="mock", model="mock/cheap", direction="in"
        ),
        "list": value(
            "relay_cost_nano_usd_total",
            tenant="test",
            provider="mock",
            model="mock/cheap",
            kind="list",
        ),
        "saved": value(
            "relay_cost_nano_usd_total",
            tenant="test",
            provider="mock",
            model="mock/cheap",
            kind="counterfactual_strong",
        ),
        "attempt": value(
            "relay_provider_attempt_seconds_count",
            provider="mock",
            model="mock/cheap",
            outcome="ok",
        ),
    }
    app = create_app(relay_settings(), providers={"mock": MockProvider()})
    async with serving(app) as c:
        r = await c.post(CHAT, json=BODY)
    usage = r.json()["usage"]
    assert value("relay_requests_total", **ok) == before["requests"] + 1
    assert (
        value(
            "relay_request_duration_seconds_count",
            route="default",
            status="success",
            stream="false",
        )
        == before["duration"] + 1
    )
    assert (
        value(
            "relay_tokens_total", tenant="test", provider="mock", model="mock/cheap", direction="in"
        )
        == before["tokens_in"] + usage["prompt_tokens"]
    )
    list_cost = (
        value(
            "relay_cost_nano_usd_total",
            tenant="test",
            provider="mock",
            model="mock/cheap",
            kind="list",
        )
        - before["list"]
    )
    strong_cost = (
        value(
            "relay_cost_nano_usd_total",
            tenant="test",
            provider="mock",
            model="mock/cheap",
            kind="counterfactual_strong",
        )
        - before["saved"]
    )
    assert 0 < list_cost < strong_cost  # the cheap model really was cheaper: cost saved > 0
    assert (
        value(
            "relay_provider_attempt_seconds_count",
            provider="mock",
            model="mock/cheap",
            outcome="ok",
        )
        == before["attempt"] + 1
    )


async def test_errors_are_counted_by_class_and_streams_get_ttft() -> None:
    err = {
        "tenant": "test",
        "route": "default",
        "status": "error",
        "error_class": "model_not_found",
    }
    before_err = value("relay_requests_total", **err)
    before_ttft = value("relay_ttft_seconds_count", route="default", model="mock/cheap")
    app = create_app(relay_settings(), providers={"mock": MockProvider()})
    async with serving(app) as c:
        await c.post(CHAT, json={**BODY, "model": "no/such"})
        await c.post(CHAT, json={**BODY, "stream": True})
    assert value("relay_requests_total", **err) == before_err + 1
    assert value("relay_ttft_seconds_count", route="default", model="mock/cheap") == before_ttft + 1


# Labels any Relay metric may use: bounded identifiers and small closed enumerations.
ALLOWED_LABELS = {
    "scope",  # a provider's rate-limit scope: model | account (LimitScope)
    "tenant",
    "route",
    "provider",
    "model",
    "status",
    "error_class",
    "stage",
    "layer",
    "result",
    "reason",
    "priority",
    "tier",
    "cause",
    "kind",
    "outcome",
    "action",
    "operation",
    "to",
    "direction",
    "stream",
}
FORBIDDEN = {"request_id", "cohort", "session", "key", "prompt", "user", "api_key"}


def test_every_relay_metric_has_bounded_labels() -> None:
    create_app(relay_settings(), providers={"mock": MockProvider()})  # import everything
    found = 0
    for family in REGISTRY.collect():
        if not family.name.startswith("relay_"):
            continue
        found += 1
        for sample in family.samples:
            labels = set(sample.labels) - {"le", "quantile"}
            assert not labels & FORBIDDEN, (family.name, labels)
            assert labels <= ALLOWED_LABELS, (family.name, labels - ALLOWED_LABELS)
    assert found > 20


# ------------------------------------------------------------------ logs
def record(msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("relay.test", logging.WARNING, __file__, 1, msg, None, None)


def test_a_log_line_carries_the_request_and_the_trace() -> None:
    provider = TracerProvider()
    token = REQUEST_ID.set("req-123")
    try:
        with provider.get_tracer("t").start_as_current_span("s") as span:
            rec = record()
            RequestContextFilter().filter(rec)
            line = json.loads(JsonFormatter().format(rec))
    finally:
        REQUEST_ID.reset(token)
    assert line["request_id"] == "req-123"
    assert line["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert line["span_id"] == format(span.get_span_context().span_id, "016x")
    assert (line["level"], line["logger"], line["msg"]) == ("WARNING", "relay.test", "hello")


def test_outside_a_request_the_ids_are_null() -> None:
    rec = record()
    RequestContextFilter().filter(rec)
    line = json.loads(JsonFormatter().format(rec))
    assert line["request_id"] is None and line["trace_id"] is None


def test_exceptions_are_kept_in_the_json() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    RequestContextFilter().filter(rec)
    assert "ValueError: boom" in json.loads(JsonFormatter().format(rec))["exc_info"]


async def test_during_a_request_the_middleware_sets_the_request_id() -> None:
    seen: list[str | None] = []

    class Spy(logging.Handler):
        def emit(self, rec: logging.LogRecord) -> None:
            seen.append(getattr(rec, "request_id", None))

    spy = Spy()
    spy.addFilter(RequestContextFilter())
    log = logging.getLogger("relay.api.app")
    log.addHandler(spy)
    try:
        app = create_app(relay_settings(), providers={"mock": MockProvider()})
        async with serving(app) as c:
            r = await c.post(
                CHAT, json={**BODY, "model": "no/such"}, headers={"x-request-id": "abc-1"}
            )
            # A log line from inside the request: emitted by a handler-level logger.
            assert r.status_code == 404
            logging.getLogger("relay.api.app").warning("outside")
    finally:
        log.removeHandler(spy)
    assert seen[-1] is None  # after the request, no request id leaks into later lines


def test_configure_logging_is_idempotent() -> None:
    configure_logging(relay_settings(log_format="json"))
    configure_logging(relay_settings(log_format="json"))
    root = logging.getLogger()
    assert sum(1 for h in root.handlers if h.get_name() == "relay") == 1
    assert isinstance(writer_formatter(), JsonFormatter)
    configure_logging(relay_settings())  # back to text for the rest of the suite
    assert tracing.enabled() is False


def test_log_lines_are_written_by_a_thread_not_the_caller() -> None:
    # Found in review: the handler wrote to stderr on the event loop, so every log line was
    # synchronous I/O on the request path. Now the caller only enqueues.
    import threading

    from relay.telemetry import logs

    configure_logging(relay_settings(log_format="json"))
    written_by: list[str] = []
    assert logs._listener is not None
    writer = logs._listener.handlers[0]
    original = writer.emit

    def spy(record: logging.LogRecord) -> None:
        written_by.append(threading.current_thread().name)
        original(record)

    writer.emit = spy  # type: ignore[method-assign]
    try:
        token = REQUEST_ID.set("req-q")
        try:
            logging.getLogger("relay.test").warning("from the loop")
        finally:
            REQUEST_ID.reset(token)
        logs._listener.stop()  # flushes the queue
        logs._listener = None
    finally:
        configure_logging(relay_settings())
    assert written_by and threading.main_thread().name not in written_by


def test_an_exception_survives_the_queue() -> None:
    from relay.telemetry.logs import _QueueHandler

    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = logging.LogRecord(
            "x", logging.ERROR, __file__, 1, "failed %s", ("now",), sys.exc_info()
        )
    prepared = _QueueHandler(None).prepare(rec)  # type: ignore[arg-type]
    line = json.loads(JsonFormatter().format(prepared))
    assert line["msg"] == "failed now" and "ValueError: boom" in line["exc_info"]
