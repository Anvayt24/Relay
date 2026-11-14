"""Refuse an oversized request body before reading it.

A pure-ASGI middleware, like the request-ID one, so it sees the body as it arrives instead of
after the framework has buffered all of it.

- **A declared length over the limit** is refused at once, without reading a byte.
- **A declared length within the limit** passes straight through: the server enforces the
  declared length, so the body cannot be longer than it said. The common case costs one
  header comparison.
- **No declared length** (chunked uploads) is read here, up to the limit, then handed to the
  app as one piece. A client that keeps sending is refused as soon as it passes the limit.

The refusal is the normal error envelope, so SDKs parse it like any other Relay error.
"""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from relay.api.errors import error_body
from relay.schemas.errors import ErrorClass


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self._max = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None:
            if declared > self._max:
                await self._refuse(send)
                return
            await self.app(scope, receive, send)
            return

        # No Content-Length: count what arrives.
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # The client went away before finishing: let the app see the same thing.
                await self.app(scope, _replay([message]), send)
                return
            piece: bytes = message.get("body", b"")
            total += len(piece)
            if total > self._max:
                await self._refuse(send)
                return
            chunks.append(piece)
            if not message.get("more_body", False):
                break
        whole: Message = {"type": "http.request", "body": b"".join(chunks), "more_body": False}
        await self.app(scope, _replay([whole], then=receive), send)

    async def _refuse(self, send: Send) -> None:
        body = json.dumps(
            error_body(
                ErrorClass.REQUEST_TOO_LARGE,
                f"the request body is larger than the {self._max} bytes this gateway accepts",
            )
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-should-retry", b"false"),  # the same body would be refused again
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None  # malformed: treat it as undeclared and count the bytes
    return None


def _replay(messages: list[Message], then: Receive | None = None) -> Receive:
    """A `receive` that yields `messages`, then whatever `then` yields (the disconnect)."""
    queue = list(messages)

    async def receive() -> Message:
        if queue:
            return queue.pop(0)
        if then is not None:
            return await then()
        return {"type": "http.disconnect"}

    return receive
