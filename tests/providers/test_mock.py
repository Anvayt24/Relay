import time

import pytest

from relay.providers.mock import MockBehavior, MockProvider
from relay.schemas.chat import ChatRequest
from relay.schemas.errors import ErrorClass, ProviderError


def _request(content: str = "hello there", max_tokens: int | None = None) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": "mock-strong-v1",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
        }
    )


async def test_echoes_last_user_message() -> None:
    provider = MockProvider()
    response = await provider.complete(_request("hello there"))

    choice = response.choices[0]
    assert choice.message.content == "mock: hello there"
    assert choice.finish_reason == "stop"
    assert response.model == "mock-strong-v1"
    assert response.id.startswith("chatcmpl-")
    assert provider.calls == 1


async def test_usage_counts_words() -> None:
    response = await MockProvider().complete(_request("hello there"))
    # Prompt: "be brief" (2) + "hello there" (2). Completion: "mock: hello there" (3).
    assert response.usage.prompt_tokens == 4
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 7


async def test_max_tokens_truncates_with_length_finish_reason() -> None:
    response = await MockProvider().complete(_request("one two three four", max_tokens=2))
    choice = response.choices[0]
    assert choice.message.content == "mock: one"
    assert choice.finish_reason == "length"
    assert response.usage.completion_tokens == 2


async def test_injected_error_raises_provider_error() -> None:
    provider = MockProvider(MockBehavior(error=ErrorClass.OVERLOADED, retry_after_s=3.0))
    with pytest.raises(ProviderError) as exc:
        await provider.complete(_request())
    assert exc.value.error_class is ErrorClass.OVERLOADED
    assert exc.value.signal is not None
    assert exc.value.signal.retry_after_s == 3.0
    assert provider.calls == 1


async def test_stream_shape() -> None:
    provider = MockProvider()
    chunks = [c async for c in provider.stream(_request("hello there"))]

    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    # Streaming and non-streaming produce the same text.
    assert text == "mock: hello there"
    assert chunks[-2].choices[0].finish_reason == "stop"
    # The usage chunk always comes last, with no choices, as in OpenAI's format.
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 3
    assert provider.open_streams == 0


async def test_stream_fails_after_k_content_chunks() -> None:
    provider = MockProvider(MockBehavior(fail_after_chunks=2))
    received: list[str] = []
    with pytest.raises(ProviderError) as exc:
        async for chunk in provider.stream(_request("one two three")):
            if chunk.has_content:
                received.append(chunk.choices[0].delta.content or "")
    assert received == ["mock:", " one"]
    assert exc.value.error_class is ErrorClass.STREAM_BROKEN


async def test_stream_error_without_fail_after_fails_immediately() -> None:
    provider = MockProvider(MockBehavior(error=ErrorClass.SERVER_ERROR))
    with pytest.raises(ProviderError):
        await anext(provider.stream(_request()))
    assert provider.open_streams == 0


async def test_aclose_runs_the_stream_cleanup() -> None:
    provider = MockProvider()
    stream = provider.stream(_request())
    await anext(stream)
    assert provider.open_streams == 1
    await stream.aclose()
    assert provider.open_streams == 0


async def test_latency_is_simulated() -> None:
    provider = MockProvider(MockBehavior(latency_s=0.1))
    started = time.monotonic()
    await provider.complete(_request())
    # Not ">= 0.1": on Windows the monotonic clock ticks in ~16 ms steps, so a 100 ms sleep
    # can measure as 94 ms. The test only needs to show the delay happened.
    assert time.monotonic() - started >= 0.08
