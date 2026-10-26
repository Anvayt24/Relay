"""Acceptance tests for streaming through the official OpenAI SDK (PRD FR-1, FR-5).

The SDK's transport here (httpx2.ASGITransport) runs the app to completion and then hands the
body over, so these tests check *what* is sent, not *when*. Timing and disconnects are covered
at the raw ASGI level in test_streaming_asgi.py.
"""

from collections.abc import AsyncIterator, Callable

import httpx2
import openai
import pytest
from fastapi import FastAPI

from relay.providers.mock import MockBehavior, MockProvider
from relay.schemas.errors import ErrorClass

MESSAGES = [{"role": "user", "content": "stream this please"}]


@pytest.fixture
async def sdk(
    request: pytest.FixtureRequest, make_app: Callable[..., FastAPI]
) -> AsyncIterator[tuple[MockProvider, openai.AsyncOpenAI]]:
    provider = MockProvider(getattr(request, "param", MockBehavior()))
    http_client = openai.DefaultAsyncHttpxClient(
        transport=httpx2.ASGITransport(app=make_app(provider))
    )
    client = openai.AsyncOpenAI(
        base_url="http://relay.test/v1", api_key="test-key", http_client=http_client
    )
    async with client:
        yield provider, client


async def test_stream_text_matches_non_stream(sdk: tuple[MockProvider, openai.AsyncOpenAI]) -> None:
    _, client = sdk
    stream = await client.chat.completions.create(
        model="relay/auto", messages=MESSAGES, stream=True
    )
    chunks = [chunk async for chunk in stream]  # iteration ends cleanly on [DONE]

    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    full = await client.chat.completions.create(model="relay/auto", messages=MESSAGES)
    assert text == full.choices[0].message.content == "mock: stream this please"
    assert all(c.model == "mock/strong" for c in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop"
    # Not requested, so not forwarded.
    assert all(c.usage is None for c in chunks)


async def test_usage_only_when_requested(sdk: tuple[MockProvider, openai.AsyncOpenAI]) -> None:
    _, client = sdk
    stream = await client.chat.completions.create(
        model="relay/cheap",
        messages=MESSAGES,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [chunk async for chunk in stream]

    last = chunks[-1]
    assert last.choices == []
    assert last.usage is not None
    assert last.usage.completion_tokens == 4
    # The `relay` extension (FR-1) rides along as an extra field; the SDK keeps it instead of
    # failing to parse. This is the "verify for each SDK you claim to support" check.
    assert last.model_extra is not None
    assert last.model_extra["relay"]["model"] == "mock/cheap"


async def test_relay_headers_on_stream(sdk: tuple[MockProvider, openai.AsyncOpenAI]) -> None:
    _, client = sdk
    raw = await client.chat.completions.with_raw_response.create(
        model="relay/cheap", messages=MESSAGES, stream=True
    )
    assert raw.headers["content-type"].startswith("text/event-stream")
    assert raw.headers["x-relay-model"] == "mock/cheap"
    assert raw.headers["x-relay-provider"] == "mock"
    assert raw.headers["x-request-id"]
    [_ async for _ in raw.parse()]


@pytest.mark.parametrize(
    ("sdk", "sdk_error", "relay_calls"),
    [
        # overloaded: Relay's own single retry (Appendix B), then the error.
        (MockBehavior(error=ErrorClass.OVERLOADED), openai.InternalServerError, 2),
        # The role chunk arrived, then the provider failed. Nothing had been sent to the
        # client yet, so it's still a clean HTTP error, not a broken stream.
        (
            MockBehavior(fail_after_chunks=0, error=ErrorClass.RATE_LIMITED),
            openai.RateLimitError,
            1,
        ),
    ],
    indirect=["sdk"],
)
async def test_failure_before_first_content_is_an_http_error(
    sdk: tuple[MockProvider, openai.AsyncOpenAI],
    sdk_error: type[openai.APIStatusError],
    relay_calls: int,
) -> None:
    provider, client = sdk
    with pytest.raises(sdk_error):
        await client.chat.completions.create(model="relay/auto", messages=MESSAGES, stream=True)
    # Only Relay's attempts: x-should-retry: false holds for streams too.
    assert provider.calls == relay_calls


@pytest.mark.parametrize("sdk", [MockBehavior(fail_after_chunks=2)], indirect=True)
async def test_failure_after_first_content_raises_mid_iteration(
    sdk: tuple[MockProvider, openai.AsyncOpenAI],
) -> None:
    provider, client = sdk
    stream = await client.chat.completions.create(
        model="relay/auto", messages=MESSAGES, stream=True
    )

    received: list[str] = []
    with pytest.raises(openai.APIError) as exc:
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                received.append(chunk.choices[0].delta.content)

    # The client got the first part, then a clear error, never a silently truncated answer.
    assert received == ["mock:", " stream"]
    assert "stream broken" in exc.value.message
    assert provider.calls == 1
