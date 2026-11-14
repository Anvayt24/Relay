"""OpenTelemetry tracing, hand-written (PRD FR-8; ADR 0025).

One trace per request, shaped like the pipeline:

    relay.request                      the request-ID middleware: the whole request,
    ├── relay.auth                     including a stream's body until it ends
    ├── relay.stage.admission
    ├── relay.stage.cache
    ├── relay.stage.route
    ├── relay.stage.reservation
    └── relay.stage.provider           events: candidates skipped without a call
        ├── relay.provider.attempt     hop 0, retry 0  (error_class=server_error)
        ├── relay.provider.attempt     hop 0, retry 1
        └── relay.provider.attempt     hop 1           (the fallback that served)

The request span gets the request's outcome at the end (``annotate``, called from
``LogStage.finalize``): cache status, router score and reason, model, tier, policy version,
canary arm, fallback hops, degraded.

**Off unless configured.** ``setup_tracing`` installs a provider only when
``RELAY_OTEL_ENDPOINT`` is set. Without one, ``enabled()`` is False, the middleware opens no
span, and every other span call goes to OTel's no-op tracer: nothing is recorded or exported.

**Export never blocks.** ``BatchSpanProcessor`` queues finished spans (bounded) and a
background thread sends them in batches over OTLP gRPC; when the queue is full, new spans are
dropped. A down collector costs dropped spans, never a slower or failed request.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, Status, StatusCode

from relay import __version__
from relay.schemas.errors import ProviderError, RelayError

if TYPE_CHECKING:
    from relay.pipeline.context import RequestContext
    from relay.pipeline.stage import Outcome
    from relay.settings import Settings

logger = logging.getLogger(__name__)

_provider: TracerProvider | None = None


def enabled() -> bool:
    return _provider is not None


def tracer() -> trace.Tracer:
    """Relay's tracer: the installed provider's, else OTel's no-op one."""
    if _provider is not None:
        return _provider.get_tracer("relay", __version__)
    return trace.get_tracer("relay", __version__)


def use_provider(provider: TracerProvider | None) -> None:
    """Install (or, with None, remove) the provider. Tests use it with an in-memory exporter."""
    global _provider
    _provider = provider


def setup_tracing(settings: "Settings") -> TracerProvider | None:
    """Build and install the provider if an OTLP endpoint is configured."""
    if not settings.otel_endpoint:
        return None
    # Imported here: the gRPC exporter pulls in grpcio, which nothing else needs.
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "relay",
                "service.version": __version__,
                "deployment.environment": settings.env,
            }
        ),
        # Respect the caller's decision when a traceparent arrives; otherwise sample a share.
        sampler=ParentBased(TraceIdRatioBased(settings.otel_sample_ratio)),
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True),
            max_queue_size=4096,  # bounded: beyond this, spans are dropped, never waited for
            max_export_batch_size=512,
            schedule_delay_millis=1000,
        )
    )
    use_provider(provider)
    logger.info(
        "tracing on: OTLP to %s, sample ratio %s",
        settings.otel_endpoint,
        settings.otel_sample_ratio,
    )
    return provider


def shutdown(timeout_s: float = 5.0) -> None:
    """Flush queued spans (bounded) and stop the exporter thread."""
    if _provider is not None:
        _provider.force_flush(int(timeout_s * 1000))
        _provider.shutdown()


# ------------------------------------------------------------------ spans
@contextmanager
def attempt_span(
    *, provider: str, model: str, hop: int, retry: int, left_because: str | None
) -> Iterator[Span]:
    """One upstream call (``relay.provider.attempt``). A ``ProviderError`` marks the span as
    an error with its class and is re-raised unchanged."""
    attributes: dict[str, str | int | bool] = {
        "relay.provider": provider,
        "relay.model": model,
        "relay.hop": hop,
        "relay.retry": retry,
    }
    if hop > 0 and left_because:
        attributes["relay.fallback.reason"] = left_because
    with tracer().start_as_current_span(
        "relay.provider.attempt",
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except ProviderError as e:
            mark_error(span, e.error_class.value)
            raise


@contextmanager
def error_span(name: str) -> Iterator[Span]:
    """A span that records a ``RelayError``'s class (and other exceptions) as its error."""
    with tracer().start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except RelayError as e:
            mark_error(span, e.error_class.value)
            raise
        except BaseException as e:
            if not isinstance(e, GeneratorExit):
                mark_error(span, type(e).__name__)
            raise


def mark_error(span: Span, error_class: str) -> None:
    span.set_attribute("relay.error_class", error_class)
    span.set_status(Status(StatusCode.ERROR, error_class))


def skipped(model: str, provider: str, hop: int, reason: str | None) -> None:
    """A candidate the dispatcher passed over without calling: an event on the current span,
    so a trace answers "why did this request end up on that model?"."""
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(
            "relay.candidate.skipped",
            {
                "relay.model": model,
                "relay.provider": provider,
                "relay.hop": hop,
                "relay.reason": reason or "unknown",
            },
        )


def annotate(ctx: "RequestContext", outcome: "Outcome") -> None:
    """The request's outcome on its root span (FR-8's attribute list), set once at the end."""
    span = ctx.span if ctx.span is not None else trace.get_current_span()
    if not span.is_recording():
        return
    attrs: dict[str, str | int | float | bool] = {
        "relay.tenant": ctx.tenant.id,
        "relay.route": ctx.route,
        "relay.stream": ctx.stream,
        "relay.outcome": outcome.status,
        "relay.policy_version": ctx.policy_version,
        "relay.fallback_hops": ctx.fallback_hops,
        "relay.degraded": ctx.degraded,
    }
    if ctx.cache is not None:
        attrs["relay.cache.status"] = ctx.cache.status
        if ctx.cache.layer:
            attrs["relay.cache.layer"] = ctx.cache.layer
    if ctx.router_reason is not None:
        attrs["relay.router.reason"] = ctx.router_reason
    if ctx.router_score is not None:
        attrs["relay.router.score"] = ctx.router_score
    if ctx.decision is not None:
        attrs["relay.tier"] = ctx.decision.tier
    served = ctx.served_by
    if served is not None:
        attrs["relay.provider"], attrs["relay.model"] = served
    if ctx.canary_arm is not None:
        attrs["relay.canary_arm"] = ctx.canary_arm
    if outcome.error_class is not None:
        mark_error(span, outcome.error_class.value)
    span.set_attributes(attrs)
