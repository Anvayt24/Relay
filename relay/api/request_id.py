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
"""

import re
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

HEADER = "X-Request-ID"
_RAW_HEADER = HEADER.lower().encode("latin-1")  # ASGI header names are lowercase bytes

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

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[HEADER] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)
