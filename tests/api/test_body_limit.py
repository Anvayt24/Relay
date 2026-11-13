"""A request body over RELAY_MAX_BODY_BYTES is refused with 413 before it is read."""

import json
from typing import Any

import httpx

from relay.api.app import create_app
from relay.api.body_limit import BodyLimitMiddleware
from relay.providers.mock import MockProvider

from ..conftest import test_settings as relay_settings  # type: ignore[import-not-found]

CHAT = "/v1/chat/completions"


def _body(content: str) -> dict[str, Any]:
    return {"model": "relay/cheap", "messages": [{"role": "user", "content": content}]}


async def _client(**settings: Any) -> httpx.AsyncClient:
    app = create_app(relay_settings(**settings), providers={"mock": MockProvider()})
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://relay.test")


async def test_a_body_over_the_limit_is_413_in_relays_error_format() -> None:
    c = await _client(max_body_bytes=2000)
    async with c:
        r = await c.post(CHAT, json=_body("x" * 5000))
    assert r.status_code == 413
    error = r.json()["error"]
    assert error["code"] == "request_too_large" and "2000" in error["message"]
    assert r.headers["x-should-retry"] == "false"  # sending it again cannot help
    assert "x-request-id" in r.headers  # the middleware sits inside the request-ID one


async def test_a_body_within_the_limit_is_served() -> None:
    c = await _client(max_body_bytes=2000)
    async with c:
        r = await c.post(CHAT, json=_body("x" * 100))
    assert r.status_code == 200


async def test_the_default_limit_is_one_mebibyte() -> None:
    assert relay_settings().max_body_bytes == 1_048_576


# ------------------------------------------------------------------ raw ASGI: chunked bodies
class Recorder:
    """An ASGI app that reads its whole body, and a `send` that remembers the response."""

    def __init__(self) -> None:
        self.body: bytes | None = None
        self.sent: list[dict[str, Any]] = []

    async def app(self, scope: Any, receive: Any, send: Any) -> None:
        chunks = []
        while True:
            message = await receive()
            chunks.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        self.body = b"".join(chunks)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)


def _scope(headers: list[tuple[bytes, bytes]]) -> dict[str, Any]:
    return {"type": "http", "method": "POST", "path": CHAT, "headers": headers}


def _receiver(*chunks: bytes) -> Any:
    messages = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ]

    async def receive() -> dict[str, Any]:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    receive.remaining = messages  # type: ignore[attr-defined]
    return receive


async def test_a_declared_length_over_the_limit_is_refused_without_reading_the_body() -> None:
    rec = Recorder()
    receive = _receiver(b"x" * 10)
    middleware = BodyLimitMiddleware(rec.app, max_bytes=5)
    await middleware(_scope([(b"content-length", b"999")]), receive, rec.send)
    assert rec.body is None  # the app never ran
    assert rec.sent[0]["status"] == 413
    assert len(receive.remaining) == 1  # and not one byte of the body was read


async def test_a_chunked_body_over_the_limit_is_refused() -> None:
    rec = Recorder()
    middleware = BodyLimitMiddleware(rec.app, max_bytes=10)
    # No content-length: the size is only known by counting what arrives.
    await middleware(_scope([]), _receiver(b"x" * 6, b"x" * 6), rec.send)
    assert rec.body is None
    assert rec.sent[0]["status"] == 413
    assert json.loads(rec.sent[1]["body"])["error"]["code"] == "request_too_large"


async def test_a_chunked_body_within_the_limit_reaches_the_app_intact() -> None:
    rec = Recorder()
    middleware = BodyLimitMiddleware(rec.app, max_bytes=100)
    await middleware(_scope([]), _receiver(b"abc", b"def"), rec.send)
    assert rec.body == b"abcdef"
    assert rec.sent[0]["status"] == 200


async def test_a_body_exactly_at_the_limit_is_allowed() -> None:
    rec = Recorder()
    middleware = BodyLimitMiddleware(rec.app, max_bytes=6)
    await middleware(_scope([]), _receiver(b"abc", b"def"), rec.send)
    assert rec.body == b"abcdef" and rec.sent[0]["status"] == 200


async def test_websockets_and_lifespan_pass_straight_through() -> None:
    calls: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        calls.append(scope["type"])

    middleware = BodyLimitMiddleware(app, max_bytes=1)
    await middleware({"type": "lifespan"}, _receiver(), Recorder().send)
    assert calls == ["lifespan"]
