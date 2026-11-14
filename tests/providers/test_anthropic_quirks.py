"""Anthropic-specific translation details, fed as exact bytes (httpx.MockTransport)."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from relay.providers.anthropic import AnthropicAdapter
from relay.schemas.chat import ChatRequest
from relay.schemas.errors import ErrorClass, ProviderError

Handler = Callable[[httpx.Request], httpx.Response]


def _adapter(handler: Handler) -> AnthropicAdapter:
    return AnthropicAdapter(
        base_url="http://anthropic.test", api_key="k", transport=httpx.MockTransport(handler)
    )


def _request(**extra: Any) -> ChatRequest:
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    return ChatRequest.model_validate({"model": "claude", "messages": messages, **extra})


def _message(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 1},
        **overrides,
    }


def _events(*events: tuple[str, dict[str, Any]]) -> Handler:
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=body.encode()
    )


START = (
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "model": "claude",
            "usage": {"input_tokens": 3, "output_tokens": 0},
        },
    },
)
DELTA = (
    "content_block_delta",
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
)
END = (
    "message_delta",
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
)
STOP = ("message_stop", {"type": "message_stop"})


async def test_request_translation() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_message())

    adapter = _adapter(handler)
    await adapter.complete(_request(stop=["END"], temperature=0.5))
    await adapter.aclose()
    body = json.loads(seen[0].content)
    assert body["system"] == "be brief"  # top-level, not a message
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 1024  # required by Anthropic: filled with the default
    assert body["stop_sequences"] == ["END"]
    assert seen[0].headers["x-api-key"] == "k"
    assert seen[0].headers["anthropic-version"] == "2023-06-01"


@pytest.mark.parametrize(
    ("stop_reason", "finish"),
    [("end_turn", "stop"), ("stop_sequence", "stop"), ("max_tokens", "length")],
)
async def test_stop_reason_mapping(stop_reason: str, finish: str) -> None:
    adapter = _adapter(lambda r: httpx.Response(200, json=_message(stop_reason=stop_reason)))
    response = await adapter.complete(_request())
    await adapter.aclose()
    assert response.choices[0].finish_reason == finish


async def test_cache_usage_is_summed_into_prompt_tokens() -> None:
    # Anthropic reports cache reads/writes separately from input_tokens; OpenAI's prompt_tokens
    # includes them. Relay normalizes to OpenAI's meaning.
    usage = {
        "input_tokens": 10,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 50,
        "output_tokens": 5,
    }
    adapter = _adapter(lambda r: httpx.Response(200, json=_message(usage=usage)))
    response = await adapter.complete(_request())
    await adapter.aclose()
    assert response.usage.prompt_tokens == 160
    assert response.usage.prompt_tokens_details is not None
    assert response.usage.prompt_tokens_details.cached_tokens == 100
    assert response.usage.cache_write_tokens == 50
    assert "cache_write_tokens" not in response.model_dump()  # never sent to clients


async def test_stream_usage_is_stitched_from_two_events() -> None:
    adapter = _adapter(_events(START, ("ping", {"type": "ping"}), DELTA, END, STOP))
    chunks = [c async for c in adapter.stream(_request())]
    await adapter.aclose()
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].usage is not None
    assert (chunks[-1].usage.prompt_tokens, chunks[-1].usage.completion_tokens) == (3, 1)


async def test_stream_without_message_stop_is_broken() -> None:
    adapter = _adapter(_events(START, DELTA))
    with pytest.raises(ProviderError) as exc:
        _ = [c async for c in adapter.stream(_request())]
    await adapter.aclose()
    assert exc.value.error_class is ErrorClass.STREAM_BROKEN


async def test_wrong_shaped_event_after_content_is_stream_broken() -> None:
    bad_usage = ("message_delta", {"type": "message_delta", "usage": {"output_tokens": "many"}})
    adapter = _adapter(_events(START, DELTA, bad_usage))
    with pytest.raises(ProviderError) as exc:
        _ = [c async for c in adapter.stream(_request())]
    await adapter.aclose()
    assert exc.value.error_class is ErrorClass.STREAM_BROKEN


async def test_error_event_before_content_is_classified() -> None:
    error = (
        "error",
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    adapter = _adapter(_events(START, error))
    with pytest.raises(ProviderError) as exc:
        _ = [c async for c in adapter.stream(_request())]
    await adapter.aclose()
    assert exc.value.error_class is ErrorClass.OVERLOADED


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (
            400,
            "Your credit balance is too low to access the Anthropic API.",
            ErrorClass.QUOTA_EXHAUSTED,
        ),
        (
            400,
            "prompt is too long: 250000 tokens > 200000 maximum",
            ErrorClass.CONTEXT_LENGTH_EXCEEDED,
        ),
        (529, "Overloaded", ErrorClass.OVERLOADED),
        (403, "permission denied", ErrorClass.AUTH),
    ],
)
async def test_error_classification(status: int, message: str, expected: ErrorClass) -> None:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    adapter = _adapter(lambda r: httpx.Response(status, json=body))
    with pytest.raises(ProviderError) as exc:
        await adapter.complete(_request())
    await adapter.aclose()
    assert exc.value.error_class is expected


async def test_quota_error_carries_a_pause_signal() -> None:
    # Found in review: the Anthropic adapter's own copy of the signal code left out quota
    # errors, so admission never paused the scope that ran out. Both adapters share it now.
    body = {"type": "error", "error": {"type": "invalid_request_error",
                                       "message": "Your credit balance is too low."}}  # fmt: skip
    adapter = _adapter(lambda r: httpx.Response(400, json=body, headers={"retry-after": "30"}))
    with pytest.raises(ProviderError) as exc:
        await adapter.complete(_request())
    await adapter.aclose()
    assert exc.value.error_class is ErrorClass.QUOTA_EXHAUSTED
    assert exc.value.signal is not None
    assert (exc.value.signal.retry_after_s, exc.value.signal.limit_scope) == (30.0, "model")


# ---------------------------------------------------------------- thinking switch
async def test_thinking_is_switched_off_only_for_the_models_configured_that_way() -> None:
    # Claude Sonnet 5.5 thinks by default, billed as output; "between_tools" turns it off.
    # Another model on the same provider (a fallback, say) must be left as it was.
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_message())

    adapter = AnthropicAdapter(
        base_url="http://anthropic.test",
        api_key="k",
        transport=httpx.MockTransport(handler),
        thinking_off=frozenset({"claude"}),
    )
    await adapter.complete(_request())
    await adapter.complete(ChatRequest.model_validate(_request().model_dump() | {"model": "other"}))
    await adapter.aclose()
    assert seen[0]["thinking"] == {"type": "between_tools"}
    assert "thinking" not in seen[1]
