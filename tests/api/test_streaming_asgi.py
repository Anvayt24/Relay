"""Streaming tests at the raw ASGI level, below any HTTP client.

httpx's ASGITransport buffers the whole response, so it can't show *when* things are sent or
simulate a client that disappears. Here the test plays the server's part: it hands the app a
``scope``, a ``receive`` and a ``send`` it controls, and records every message. That makes it
possible to:
- check that the status and headers go out together with the first content;
- break the connection after k chunks, in each of the three ways it happens for real.

Every test runs inside the app's lifespan, so the LogWriter is running. After the lifespan
exits (which flushes the writer), the MemorySink shows how the request was finalized.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.requests import ClientDisconnect
from starlette.types import Message

from relay.providers.mock import MockBehavior, MockProvider
from relay.telemetry.log_writer import MemorySink

BODY = {
    "model": "relay/auto",
    "messages": [{"role": "user", "content": "a b c d e f g h"}],
    "stream": True,
}


@dataclass
class Call:
    sent: list[Message] = field(default_factory=list)
    # Lets a test wait until the stream is visibly under way, without polling.
    three_bodies_sent: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def start(self) -> Message:
        assert self.sent[0]["type"] == "http.response.start"
        return self.sent[0]

    @property
    def bodies(self) -> list[bytes]:
        return [m["body"] for m in self.sent if m["type"] == "http.response.body" and m["body"]]

    @property
    def events(self) -> list[Any]:
        out: list[Any] = []
        for body in self.bodies:
            data = body.decode().removeprefix("data: ").strip()
            out.append(data if data == "[DONE]" else json.loads(data))
        return out


def _scope(spec_version: str) -> dict[str, Any]:
    raw = json.dumps(BODY).encode()
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"relay.test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("relay.test", 80),
    }


async def call_app(
    app: FastAPI,
    *,
    spec_version: str = "2.3",
    send_fails_after: int | None = None,
    disconnect_after: int | None = None,
    call: Call | None = None,
) -> Call:
    """Run one request. ``send_fails_after=k``: the k+1-th body send raises OSError, which is
    what a server does when the socket is gone (ASGI 2.4). ``disconnect_after=k``: after k
    bodies, ``receive()`` returns ``http.disconnect`` (ASGI 2.3, what uvicorn uses)."""
    call = call or Call()
    body = json.dumps(BODY).encode()
    request_sent = False
    disconnected = asyncio.Event()

    async def receive() -> Message:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        is_body = message["type"] == "http.response.body"
        if is_body and send_fails_after is not None and len(call.bodies) >= send_fails_after:
            raise OSError("connection reset by peer")
        call.sent.append(message)
        if is_body and len(call.bodies) >= 3:
            call.three_bodies_sent.set()
        if disconnect_after is not None and len(call.bodies) >= disconnect_after:
            disconnected.set()

    await app(_scope(spec_version), receive, send)
    return call


@asynccontextmanager
async def running(app: FastAPI) -> AsyncIterator[None]:
    async with app.router.lifespan_context(app):
        yield


@pytest.fixture
def build(
    make_app: Callable[..., FastAPI],
) -> Callable[..., tuple[FastAPI, MockProvider, MemorySink]]:
    def _build(**behavior: Any) -> tuple[FastAPI, MockProvider, MemorySink]:
        provider = MockProvider(MockBehavior(**behavior))
        sink = MemorySink()
        app = make_app(provider, sink=sink, stream_idle_timeout_s=0.2, first_token_timeout_s=1.0)
        return app, provider, sink

    return _build


async def test_headers_are_committed_with_the_first_content(build: Callable[..., Any]) -> None:
    app, provider, sink = build()
    async with running(app):
        call = await call_app(app)

    assert call.start["status"] == 200
    headers = dict(call.start["headers"])
    assert headers[b"content-type"].startswith(b"text/event-stream")
    assert headers[b"x-relay-model"] == b"mock/strong"
    # The role chunk and first content chunk are the first two events after the commit.
    first, second = call.events[:2]
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert second["choices"][0]["delta"]["content"] == "mock:"
    assert call.events[-1] == "[DONE]"
    assert provider.open_streams == 0
    [row] = sink.rows
    assert row.status == "success"
    assert row.stream is True
    assert row.ttft_ms is not None
    assert row.tokens_out == 9


async def test_failure_before_first_content_never_commits_200(build: Callable[..., Any]) -> None:
    app, provider, sink = build(fail_after_chunks=0)
    async with running(app):
        call = await call_app(app)

    # The role chunk was held back, so Relay could still answer with a real error status.
    assert call.start["status"] == 502
    assert b"text/event-stream" not in dict(call.start["headers"]).get(b"content-type", b"")
    assert provider.open_streams == 0
    [row] = sink.rows
    assert row.status == "error"


async def test_idle_upstream_breaks_the_stream_without_done(build: Callable[..., Any]) -> None:
    app, provider, sink = build(stall_after_chunks=3)
    async with running(app):
        call = await call_app(app)

    assert call.start["status"] == 200
    *chunks, last = call.events
    assert last["error"]["type"] == "stream_broken"
    assert "[DONE]" not in call.events
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "mock: a b"
    assert provider.open_streams == 0
    [row] = sink.rows
    assert row.status == "failed_midstream"
    assert row.error_class == "stream_broken"


async def test_send_failure_closes_upstream_and_finalizes_cancelled(
    build: Callable[..., Any],
) -> None:
    app, provider, sink = build()
    async with running(app):
        # Starlette turns the OSError into ClientDisconnect and lets it reach the server,
        # which is what tells the server to drop the connection.
        with pytest.raises(ClientDisconnect):
            await call_app(app, spec_version="2.4", send_fails_after=3)

    assert provider.open_streams == 0  # closed now, not left to the garbage collector
    [row] = sink.rows
    assert row.status == "cancelled"


async def test_http_disconnect_closes_upstream_and_finalizes_cancelled(
    build: Callable[..., Any],
) -> None:
    # Uvicorn's path (ASGI 2.3): Starlette listens for http.disconnect and *cancels* the task
    # that is streaming. Slow chunks keep the stream mid-flight when that happens.
    app, provider, sink = build(chunk_delay_s=0.05)
    async with running(app):
        call = await call_app(app, spec_version="2.3", disconnect_after=3)

    assert len(call.bodies) < 12  # it really stopped early
    assert provider.open_streams == 0
    [row] = sink.rows
    assert row.status == "cancelled"


async def test_cancelled_request_task_closes_upstream_and_finalizes_cancelled(
    build: Callable[..., Any],
) -> None:
    # What the server does on a hard shutdown, or what a timeout above the app would do.
    app, provider, sink = build(chunk_delay_s=0.05)
    async with running(app):
        call = Call()
        task = asyncio.create_task(call_app(app, call=call))
        await call.three_bodies_sent.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert provider.open_streams == 0
    [row] = sink.rows
    assert row.status == "cancelled"
