"""Acceptance test for PRD FR-1: the official OpenAI SDK works against Relay with only
``base_url`` changed, and with the SDK's retries left at their defaults.

Everything runs in-process. The SDK's HTTP client (httpx2, the SDK's own fork of httpx) gets an
ASGI transport, so the SDK's real request building, response parsing, error mapping and retry
logic all run against the real app, with no server or network.
"""

from collections.abc import AsyncIterator, Callable

import httpx2
import openai
import pytest
from fastapi import FastAPI

from relay.providers.mock import MockBehavior, MockProvider
from relay.schemas.errors import ErrorClass


@pytest.fixture
def sdk_for(
    make_app: Callable[[MockProvider], FastAPI],
) -> Callable[[MockProvider], openai.AsyncOpenAI]:
    def _sdk(provider: MockProvider) -> openai.AsyncOpenAI:
        http_client = openai.DefaultAsyncHttpxClient(
            transport=httpx2.ASGITransport(app=make_app(provider))
        )
        # max_retries is deliberately left at the SDK default (2): the test proves the SDK
        # does not retry because Relay told it not to, not because we turned retries off.
        return openai.AsyncOpenAI(
            base_url="http://relay.test/v1", api_key="test-key", http_client=http_client
        )

    return _sdk


@pytest.fixture
async def provider_and_sdk(
    request: pytest.FixtureRequest, sdk_for: Callable[[MockProvider], openai.AsyncOpenAI]
) -> AsyncIterator[tuple[MockProvider, openai.AsyncOpenAI]]:
    behavior: MockBehavior = getattr(request, "param", MockBehavior())
    provider = MockProvider(behavior)
    sdk = sdk_for(provider)
    async with sdk:
        yield provider, sdk


async def test_completion_through_the_sdk(
    provider_and_sdk: tuple[MockProvider, openai.AsyncOpenAI],
) -> None:
    _, sdk = provider_and_sdk
    completion = await sdk.chat.completions.create(
        model="relay/auto",
        messages=[{"role": "user", "content": "hello from the sdk"}],
        max_completion_tokens=100,  # newer SDKs send this name; Relay folds it into max_tokens
    )

    assert completion.choices[0].message.content == "mock: hello from the sdk"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.model == "mock/strong"
    assert completion.usage is not None
    assert completion.usage.completion_tokens == 5


async def test_raw_response_exposes_relay_headers(
    provider_and_sdk: tuple[MockProvider, openai.AsyncOpenAI],
) -> None:
    _, sdk = provider_and_sdk
    raw = await sdk.chat.completions.with_raw_response.create(
        model="relay/cheap", messages=[{"role": "user", "content": "hi"}]
    )
    assert raw.headers["x-relay-model"] == "mock/cheap"
    assert raw.headers["x-request-id"]


@pytest.mark.parametrize(
    ("provider_and_sdk", "sdk_error"),
    [
        (MockBehavior(error=ErrorClass.SERVER_ERROR), openai.InternalServerError),
        (MockBehavior(error=ErrorClass.OVERLOADED), openai.InternalServerError),
        (MockBehavior(error=ErrorClass.RATE_LIMITED, retry_after_s=1), openai.RateLimitError),
    ],
    indirect=["provider_and_sdk"],
)
async def test_sdk_does_not_retry_errors_relay_returns(
    provider_and_sdk: tuple[MockProvider, openai.AsyncOpenAI],
    sdk_error: type[openai.APIStatusError],
) -> None:
    provider, sdk = provider_and_sdk

    with pytest.raises(sdk_error) as exc:
        await sdk.chat.completions.create(
            model="relay/auto", messages=[{"role": "user", "content": "hi"}]
        )

    # Without `x-should-retry: false` the SDK would retry 429/5xx twice by default, so the
    # provider would see 3 calls. Relay is the single retry owner (PRD FR-1).
    assert provider.calls == 1
    body = exc.value.body
    assert isinstance(body, dict)
    assert body["type"] == exc.value.code


async def test_unknown_model_is_not_found_error(
    provider_and_sdk: tuple[MockProvider, openai.AsyncOpenAI],
) -> None:
    _, sdk = provider_and_sdk
    with pytest.raises(openai.NotFoundError):
        await sdk.chat.completions.create(
            model="nope/model", messages=[{"role": "user", "content": "hi"}]
        )


async def test_models_list(provider_and_sdk: tuple[MockProvider, openai.AsyncOpenAI]) -> None:
    _, sdk = provider_and_sdk
    models = [m.id async for m in sdk.models.list()]
    assert "relay/auto" in models
    assert "mock/cheap" in models
