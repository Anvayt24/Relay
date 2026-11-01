"""Request-ID middleware (PRD FR-1, FR-8).

This is a pure ASGI middleware: a class with ``__call__(scope, receive, send)``. It is
deliberately not Starlette's ``BaseHTTPMiddleware``, which runs the endpoint in a separate task
and wraps the response body. That has historically broken streaming responses and context
variables, and Phase 2 streams. ASGI is also the protocol everything else here is built on, so
seeing it raw is worth it:

- ``scope``: a dict describing the connection (type, path, headers...);
- ``receive()``: awaits the next event from the client (request body chunks, disconnect);
- ``send(message)``: emits an event to the client. A response is one
  ``http.response.start`` (status + headers) followed by ``http.response.body`` events.

To add a header to every response, wrap ``send`` and edit the start message as it goes past.

**Phase 14:** the same place opens the request's root span (``relay.request``) when tracing is
on, honouring an incoming W3C ``traceparent``, and sets the request-ID contextvar that log
lines read (``relay/telemetry/logs.py``). Because this middleware wraps the whole app, and a
streaming response sends its body inside that call, the span covers a stream until it ends.
"""

import re
import uuid

from opentelemetry.propagate import extract
from opentelemetry.trace import SpanKind, Status, StatusCode
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from relay.telemetry import tracing
from relay.telemetry.logs import REQUEST_ID

HEADER = "X-Request-ID"
_RAW_HEADER = HEADER.lower().encode("latin-1")  # ASGI header names are lowercase bytes

# Probes and scrapes are not traced: a healthcheck every 5 s and a Prometheus scrape every 15 s
# would otherwise fill the trace store with noise and spend export capacity on nothing.
UNTRACED_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})

# The ID ends up in log lines and response headers. Anything outside this charset could
# forge log entries ("abc\nERROR fake line") or split headers, so it is replaced.
_VALID_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def resolve_request_id(raw: str | None) -> str:
    if raw is not None and _VALID_ID.fullmatch(raw):
        return raw
    return uuid.uuid4().hex


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # e.g. "lifespan" events have no headers
            await self.app(scope, receive, send)
            return

        raw = next(
            (value.decode("latin-1") for name, value in scope["headers"] if name == _RAW_HEADER),
            None,
        )
        request_id = resolve_request_id(raw)
        # scope["state"] is what Starlette exposes as request.state, so handlers read
        # request.state.request_id. This mutates the shared scope dict, which is also why
        # the 500 handler (outside this middleware) can see it.
        scope.setdefault("state", {})["request_id"] = request_id

        token = REQUEST_ID.set(request_id)
        try:
            if not tracing.enabled() or scope.get("path") in UNTRACED_PATHS:

                async def send_with_request_id(message: Message) -> None:
                    if message["type"] == "http.response.start":
                        MutableHeaders(scope=message)[HEADER] = request_id
                    await send(message)

                await self.app(scope, receive, send_with_request_id)
                return
            await self._traced(scope, receive, send, request_id)
        finally:
            REQUEST_ID.reset(token)

    async def _traced(self, scope: Scope, receive: Receive, send: Send, request_id: str) -> None:
        carrier = {
            name.decode("latin-1"): value.decode("latin-1")
            for name, value in scope["headers"]
            if name in (b"traceparent", b"tracestate")
        }
        with tracing.tracer().start_as_current_span(
            "relay.request",
            context=extract(carrier),
            kind=SpanKind.SERVER,
            attributes={
                "relay.request_id": request_id,
                "http.request.method": scope.get("method", ""),
                "url.path": scope.get("path", ""),
            },
        ) as span:
            scope["state"]["span"] = span

            async def send_traced(message: Message) -> None:
                if message["type"] == "http.response.start":
                    MutableHeaders(scope=message)[HEADER] = request_id
                    status = int(message["status"])
                    span.set_attribute("http.response.status_code", status)
                    if status >= 500:
                        span.set_status(Status(StatusCode.ERROR))
                await send(message)

            await self.app(scope, receive, send_traced)
