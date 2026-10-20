"""Provider quirks the simulator doesn't reproduce, fed to the adapter as exact bytes.

``httpx.MockTransport`` lets a test decide precisely what the "provider" sends: a comment
line, a list-shaped error, a malformed chunk. Each case here is something a real provider
does (or has done), described in ``relay/providers/openai_compat.py``.
"""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from prometheus_client import REGISTRY

from relay.providers.openai_compat import OpenAICompatAdapter
from relay.schemas.chat import ChatChunk, ChatRequest
from relay.schemas.errors import ErrorClass, ProviderError

Handler = Callable[[httpx.Request], httpx.Response]


def _adapter(handler: Handler, **kw: Any) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        kw.pop("name", "prov"),
        base_url="http://prov.test/v1",
        api_key="k",
        transport=httpx.MockTransport(handler),
        **kw,
    )


def _request(**extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi there"}], **extra}
    )


def _chunk(content: str | None = None, finish: str | None = None, role: bool = False) -> str:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = "assistant"
    if content is not None:
        delta["content"] = content
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return "data: " + json.dumps({"id": "c1", "created": 1, "model": "m", "choices": [choice]})


USAGE = "data: " + json.dumps(
    {
        "id": "c1",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
)


def _sse(*lines: str, status: int = 200) -> Handler:
    body = "\n\n".join(lines) + "\n\n"
    return lambda request: httpx.Response(
        status, headers={"content-type": "text/event-stream"}, content=body.encode()
    )


async def _collect(adapter: OpenAICompatAdapter) -> list[ChatChunk]:
    try:
        return [c async for c in adapter.stream(_request())]
    finally:
        await adapter.aclose()


def _text(chunks: list[ChatChunk]) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)


# ---------------------------------------------------------------- streams
async def test_openrouter_comment_lines_are_ignored() -> None:
    adapter = _adapter(
        _sse(
            ": OPENROUTER PROCESSING",
            _chunk(role=True, content=""),
            _chunk("hi"),
            _chunk(finish="stop"),
            USAGE,
            "data: [DONE]",
        )
    )
    chunks = await _collect(adapter)
    assert _text(chunks) == "hi"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 3


async def test_groq_x_groq_usage_is_read() -> None:
    x_groq = "data: " + json.dumps(
        {
            "id": "c1",
            "choices": [],
            "x_groq": {"usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}},
        }
    )
    chunks = await _collect(
        _adapter(_sse(_chunk("a"), _chunk(finish="stop"), x_groq, "data: [DONE]"))
    )
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 7


async def test_missing_usage_is_synthesized_and_counted() -> None:
    before = (
        REGISTRY.get_sample_value("relay_provider_usage_missing_total", {"provider": "nousage"})
        or 0
    )
    adapter = _adapter(
        _sse(_chunk("a"), _chunk(" b"), _chunk(finish="stop"), "data: [DONE]"), name="nousage"
    )
    chunks = await _collect(adapter)
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 2  # estimated from content chunks
    after = REGISTRY.get_sample_value("relay_provider_usage_missing_total", {"provider": "nousage"})
    assert after == before + 1


async def test_finish_without_done_is_accepted() -> None:
    chunks = await _collect(_adapter(_sse(_chunk("a"), _chunk(finish="stop"), USAGE)))
    assert _text(chunks) == "a"


async def test_cut_off_stream_is_stream_broken() -> None:
    with pytest.raises(ProviderError) as exc:
        await _collect(_adapter(_sse(_chunk("a"), _chunk(" b"))))
    assert exc.value.error_class is ErrorClass.STREAM_BROKEN


async def test_empty_stream_is_server_error() -> None:
    with pytest.raises(ProviderError) as exc:
        await _collect(_adapter(_sse("data: [DONE]")))
    assert exc.value.error_class is ErrorClass.SERVER_ERROR


async def test_malformed_chunk_before_and_after_content() -> None:
    with pytest.raises(ProviderError) as before:
        await _collect(_adapter(_sse("data: {not json")))
    with pytest.raises(ProviderError) as after:
        await _collect(_adapter(_sse(_chunk("a"), "data: {not json")))
    assert before.value.error_class is ErrorClass.SERVER_ERROR
    assert after.value.error_class is ErrorClass.STREAM_BROKEN


async def test_in_band_error_event_before_content_is_classified() -> None:
    # OpenRouter reports failures inside a 200 stream. A daily free-tier cap is quota, not a
    # short rate limit: Relay must not hammer it again in a second.
    event = 'data: {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day"}}'
    with pytest.raises(ProviderError) as exc:
        await _collect(_adapter(_sse(event)))
    assert exc.value.error_class is ErrorClass.QUOTA_EXHAUSTED


async def test_in_band_error_event_after_content_is_stream_broken() -> None:
    event = 'data: {"error": {"code": 502, "message": "upstream died"}}'
    with pytest.raises(ProviderError) as exc:
        await _collect(_adapter(_sse(_chunk("a"), event)))
    assert exc.value.error_class is ErrorClass.STREAM_BROKEN


async def test_aclose_closes_the_upstream_response() -> None:
    closed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield (_chunk("a") + "\n\n").encode()
            yield (_chunk(" b") + "\n\n").encode()

        async def aclose(self) -> None:
            closed.append(True)

    adapter = _adapter(lambda r: httpx.Response(200, stream=Body()))
    stream = adapter.stream(_request())
    await anext(stream)
    await stream.aclose()  # what Relay does when its client disconnects
    await adapter.aclose()
    assert closed == [True]  # the provider connection was released, not left to the GC


# ---------------------------------------------------------------- errors
def _status(status: int, body: object, headers: dict[str, str] | None = None) -> Handler:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return lambda request: httpx.Response(status, headers=headers or {}, content=content)


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            402,
            {"error": {"message": "Insufficient credits"}},
            ErrorClass.QUOTA_EXHAUSTED,
        ),  # OpenRouter
        (429, {"error": {"code": "insufficient_quota"}}, ErrorClass.QUOTA_EXHAUSTED),  # OpenAI
        (
            429,
            [{"error": {"status": "RESOURCE_EXHAUSTED", "message": "slow down"}}],
            ErrorClass.RATE_LIMITED,
        ),  # Gemini
        (
            429,
            [
                {
                    "error": {
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "GenerateRequestsPerDay quota",
                    }
                }
            ],
            ErrorClass.QUOTA_EXHAUSTED,
        ),
        (503, {"error": {"message": "over capacity"}}, ErrorClass.OVERLOADED),
        (500, b"<html>oops</html>", ErrorClass.SERVER_ERROR),  # an HTML page from a proxy
        (
            400,
            {
                "error": {
                    "message": "Please reduce the length of the messages: "
                    "maximum context length is 8192"
                }
            },
            ErrorClass.CONTEXT_LENGTH_EXCEEDED,
        ),
        (
            403,
            {"error": {"message": "Input was flagged by moderation"}},
            ErrorClass.CONTENT_FILTERED,
        ),  # OpenRouter
        (401, {"error": {"message": "Invalid API Key"}}, ErrorClass.AUTH),
        (404, {"error": {"message": "model not found"}}, ErrorClass.MODEL_NOT_FOUND),
        (422, {"error": {"message": "bad field"}}, ErrorClass.BAD_REQUEST),
    ],
)
async def test_error_classification(status: int, body: object, expected: ErrorClass) -> None:
    adapter = _adapter(_status(status, body))
    with pytest.raises(ProviderError) as exc:
        await adapter.complete(_request())
    await adapter.aclose()
    assert exc.value.error_class is expected


async def test_rate_limit_signal_reads_headers() -> None:
    headers = {
        "retry-after-ms": "2500",
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-remaining-tokens": "1234",
    }
    adapter = _adapter(_status(429, {"error": {"message": "slow"}}, headers), limit_scope="account")
    with pytest.raises(ProviderError) as exc:
        await adapter.complete(_request())
    await adapter.aclose()
    signal = exc.value.signal
    assert signal is not None
    assert (signal.retry_after_s, signal.limit_scope) == (2.5, "account")
    assert (signal.remaining_requests, signal.remaining_tokens) == (0, 1234)


async def test_content_filter_with_no_content_is_an_error() -> None:
    body = {
        "id": "c",
        "choices": [{"message": {"content": None}, "finish_reason": "content_filter"}],
    }
    adapter = _adapter(_status(200, body))
    with pytest.raises(ProviderError) as exc:
        await adapter.complete(_request())
    await adapter.aclose()
    assert exc.value.error_class is ErrorClass.CONTENT_FILTERED


async def test_malformed_success_body_is_server_error() -> None:
    adapter = _adapter(_status(200, b"not json"))
    with pytest.raises(ProviderError) as exc:
        await adapter.complete(_request())
    await adapter.aclose()
    assert exc.value.error_class is ErrorClass.SERVER_ERROR


async def test_timeouts_and_refused_connections() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    adapter = _adapter(timeout)
    with pytest.raises(ProviderError) as exc:
        await adapter.complete(_request())
    await adapter.aclose()
    assert exc.value.error_class is ErrorClass.TIMEOUT


# ---------------------------------------------------------------- requests
async def test_request_body_uses_named_fields_and_the_providers_cap_name() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = {
            "id": "c",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, json=body)

    adapter = _adapter(handler, max_tokens_param="max_completion_tokens")
    req = ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 9,
            "temperature": 0,
            "some_unknown_field": 1,
        }
    )
    await adapter.complete(req)
    await adapter.aclose()
    assert seen[0] == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
        "max_completion_tokens": 9,
    }  # the unknown client field was not forwarded


async def test_stream_always_asks_for_usage() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _sse(_chunk("a"), _chunk(finish="stop"), USAGE)(request)

    await _collect(_adapter(handler))
    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"] == {"include_usage": True}
