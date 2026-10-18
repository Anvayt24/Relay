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


async def test_latency_is_simulated() -> None:
    provider = MockProvider(MockBehavior(latency_s=0.05))
    started = time.monotonic()
    await provider.complete(_request())
    assert time.monotonic() - started >= 0.05
