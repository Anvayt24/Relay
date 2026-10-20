import email.utils
import time
from collections.abc import AsyncIterator

import httpx
import pytest

from relay.providers.http import error_payload, retry_after, sse_events, transport_error
from relay.schemas.errors import ErrorClass


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def _events(*lines: str) -> list[tuple[str | None, str]]:
    return [(e.event, e.data) async for e in sse_events(_lines(*lines))]


async def test_sse_basic_and_named_events() -> None:
    assert await _events("data: a", "", "event: ping", "data: {}", "") == [
        (None, "a"),
        ("ping", "{}"),
    ]


async def test_sse_skips_comment_lines() -> None:
    # OpenRouter's keep-alive while a model thinks.
    assert await _events(": OPENROUTER PROCESSING", "", "data: x", "") == [(None, "x")]


async def test_sse_joins_multi_line_data_and_flushes_at_eof() -> None:
    assert await _events("data: line1", "data: line2", "", "data: tail") == [
        (None, "line1\nline2"),
        (None, "tail"),
    ]


async def test_sse_space_after_colon_is_optional() -> None:
    assert await _events("data:x", "") == [(None, "x")]


def test_retry_after_prefers_milliseconds() -> None:
    assert retry_after({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5
    assert retry_after({"retry-after": "2"}) == 2.0
    assert retry_after({}) is None
    assert retry_after({"retry-after": "garbage"}) is None


def test_retry_after_accepts_an_http_date() -> None:
    when = email.utils.formatdate(time.time() + 30, usegmt=True)
    assert retry_after({"retry-after": when}) == pytest.approx(30, abs=2)


@pytest.mark.parametrize(
    ("body", "expected_message"),
    [
        (b'{"error": {"message": "boom", "type": "server_error"}}', "boom"),  # OpenAI
        (b'[{"error": {"message": "boom", "status": "RESOURCE_EXHAUSTED"}}]', "boom"),  # Gemini
        (b'{"type": "error", "error": {"type": "overloaded_error", "message": "boom"}}', "boom"),
        (b"<html>502 Bad Gateway</html>", "<html>502 Bad Gateway</html>"),  # a proxy
    ],
)
def test_error_payload_shapes(body: bytes, expected_message: str) -> None:
    error, blob = error_payload(body)
    assert error["message"] == expected_message
    assert blob == blob.lower()


def test_transport_error_depends_on_whether_content_started() -> None:
    timeout = httpx.ReadTimeout("slow")
    assert transport_error(timeout, content_started=False).error_class is ErrorClass.TIMEOUT
    assert transport_error(timeout, content_started=True).error_class is ErrorClass.STREAM_BROKEN
    refused = httpx.ConnectError("refused")
    assert transport_error(refused, content_started=False).error_class is ErrorClass.SERVER_ERROR
