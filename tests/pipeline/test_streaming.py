from collections.abc import AsyncGenerator

import pytest

from relay.pipeline.streaming import prime
from relay.providers.mock import MockBehavior, MockProvider
from relay.schemas.chat import ChatChunk, ChatRequest, ChunkChoice, ChunkDelta
from relay.schemas.errors import ErrorClass, ProviderError


def _request(content: str = "one two three") -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": "mock-strong-v1",
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }
    )


async def test_stops_at_first_content_chunk_with_role_chunk_buffered() -> None:
    provider = MockProvider()
    primed = await prime(provider.stream(_request()), timeout_s=1, relay_model="mock/strong")

    assert len(primed.buffered) == 2
    role, first = primed.buffered
    assert role.choices[0].delta.role == "assistant"
    assert not role.has_content
    assert first.choices[0].delta.content == "mock:"
    # Relabeled: the client sees the Relay name, not the upstream ID.
    assert all(c.model == "mock/strong" for c in primed.buffered)

    # The rest is still there, untouched.
    rest = [c async for c in primed.upstream]
    assert "".join(c.choices[0].delta.content or "" for c in rest if c.choices) == " one two three"
    assert rest[-1].usage is not None


async def test_empty_completion_does_not_wait_for_content() -> None:
    async def empty() -> AsyncGenerator[ChatChunk, None]:
        yield ChatChunk(
            id="x", created=0, model="m", choices=[ChunkChoice(delta=ChunkDelta(role="assistant"))]
        )
        yield ChatChunk(
            id="x",
            created=0,
            model="m",
            choices=[ChunkChoice(delta=ChunkDelta(), finish_reason="stop")],
        )

    primed = await prime(empty(), timeout_s=1, relay_model="r")
    assert primed.buffered[-1].is_finished


async def test_immediately_ending_stream_commits_with_nothing_buffered() -> None:
    async def nothing() -> AsyncGenerator[ChatChunk, None]:
        return
        yield  # pragma: no cover  (makes this an async generator)

    primed = await prime(nothing(), timeout_s=1, relay_model="r")
    assert primed.buffered == []


async def test_slow_first_token_within_timeout_is_fine() -> None:
    provider = MockProvider(MockBehavior(first_chunk_delay_s=0.05))
    primed = await prime(provider.stream(_request()), timeout_s=1, relay_model="r")
    assert primed.buffered[-1].has_content


async def test_first_token_timeout_is_a_timeout_error_and_closes_upstream() -> None:
    provider = MockProvider(MockBehavior(first_chunk_delay_s=5))
    with pytest.raises(ProviderError) as exc:
        await prime(provider.stream(_request()), timeout_s=0.05, relay_model="r")
    assert exc.value.error_class is ErrorClass.TIMEOUT
    assert provider.open_streams == 0


async def test_error_before_first_content_is_raised_and_closes_upstream() -> None:
    # fail_after_chunks=0: the role chunk arrives, then the provider fails. Because the role
    # chunk was only buffered, this is still a clean, uncommitted failure.
    provider = MockProvider(MockBehavior(fail_after_chunks=0, error=ErrorClass.OVERLOADED))
    with pytest.raises(ProviderError) as exc:
        await prime(provider.stream(_request()), timeout_s=1, relay_model="r")
    assert exc.value.error_class is ErrorClass.OVERLOADED
    assert provider.open_streams == 0
