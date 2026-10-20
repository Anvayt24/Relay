"""A deterministic in-process provider for tests, local runs and demos.

Given the same request it always gives the same reply, with configurable failures. That makes
gateway behavior testable without network, cost or flakiness. Phase 3 adds a richer simulator
(latency distributions, error rates, ground-truth quality) in ``bench/simulator``.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal

from relay.schemas.chat import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    Choice,
    ChunkChoice,
    ChunkDelta,
    ResponseMessage,
    Usage,
)
from relay.schemas.errors import ErrorClass, ProviderError, ProviderSignal

FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class MockBehavior:
    latency_s: float = 0.0
    # Raise this class. Without fail_after_chunks it fails immediately, before any chunk.
    error: ErrorClass | None = None
    retry_after_s: float | None = None  # sent with the error, as a real 429 would
    reply_prefix: str = "mock: "

    # Streaming faults (PRD M1: slow first token, mid-stream disconnect, stall).
    first_chunk_delay_s: float = 0.0
    chunk_delay_s: float = 0.0
    # Fail after this many *content* chunks (0 = after the role chunk, before any text).
    # The class is `error`, or STREAM_BROKEN if unset.
    fail_after_chunks: int | None = None
    # Stop sending (without closing) after this many content chunks: a hung upstream.
    stall_after_chunks: int | None = None


class MockProvider:
    name = "mock"

    def __init__(self, behavior: MockBehavior | None = None) -> None:
        self.behavior = behavior or MockBehavior()
        # How many times complete()/stream() was called. Tests use it to prove nobody retried.
        self.calls = 0
        # Streams started but not yet closed. Tests use it to prove a disconnect closed the
        # upstream, instead of leaving it to the garbage collector.
        self.open_streams = 0

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.behavior.latency_s > 0:
            # asyncio.sleep, never time.sleep: a blocking sleep would freeze the event loop,
            # and with it every other request in this process.
            await asyncio.sleep(self.behavior.latency_s)
        if self.behavior.error is not None:
            raise self._error(self.behavior.error)

        words, finish_reason = self._reply(req)
        prompt_tokens = _prompt_tokens(req)
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[
                Choice(
                    message=ResponseMessage(content=" ".join(words)),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(words),
                total_tokens=prompt_tokens + len(words),
            ),
        )

    async def stream(self, req: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        self.calls += 1
        self.open_streams += 1
        try:
            # aclosing(): when *this* generator is closed, `async for` stops but does NOT close
            # the inner generator; it would stay suspended until garbage collection. The same
            # trap applies to any async generator that wraps another one.
            async with aclosing(self._stream(req)) as inner:
                async for chunk in inner:
                    yield chunk
        finally:
            # Runs on normal end, on an exception, and on aclose() (GeneratorExit).
            self.open_streams -= 1

    async def _stream(self, req: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        b = self.behavior
        if b.first_chunk_delay_s > 0:
            await asyncio.sleep(b.first_chunk_delay_s)
        if b.error is not None and b.fail_after_chunks is None:
            raise self._error(b.error)

        stream_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

        def chunk(
            delta: ChunkDelta | None = None,
            finish_reason: FinishReason | None = None,
            usage: Usage | None = None,
        ) -> ChatChunk:
            choices = (
                []
                if usage
                else [ChunkChoice(delta=delta or ChunkDelta(), finish_reason=finish_reason)]
            )
            return ChatChunk(
                id=stream_id, created=created, model=req.model, choices=choices, usage=usage
            )

        yield chunk(ChunkDelta(role="assistant", content=""))

        words, finish_reason = self._reply(req)
        for i, word in enumerate(words):
            if b.fail_after_chunks == i:
                raise self._error(b.error or ErrorClass.STREAM_BROKEN)
            if b.stall_after_chunks == i:
                await asyncio.sleep(3600)  # "forever": only the idle timeout ends this
            if i > 0 and b.chunk_delay_s > 0:
                await asyncio.sleep(b.chunk_delay_s)
            yield chunk(ChunkDelta(content=word if i == 0 else f" {word}"))

        yield chunk(finish_reason=finish_reason)
        prompt_tokens = _prompt_tokens(req)
        yield chunk(
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(words),
                total_tokens=prompt_tokens + len(words),
            )
        )

    def _reply(self, req: ChatRequest) -> tuple[list[str], FinishReason]:
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        # "Tokens" are whitespace-separated words. That is enough to exercise usage accounting
        # and max_tokens; real tokenizers arrive with admission estimates (Phase 6).
        words = f"{self.behavior.reply_prefix}{last_user}".split()
        if req.max_tokens is not None and len(words) > req.max_tokens:
            return words[: req.max_tokens], "length"
        return words, "stop"

    def _error(self, error_class: ErrorClass) -> ProviderError:
        signal = None
        if self.behavior.retry_after_s is not None:
            signal = ProviderSignal(retry_after_s=self.behavior.retry_after_s, limit_scope="model")
        return ProviderError(error_class, f"mock injected {error_class}", signal)


def _prompt_tokens(req: ChatRequest) -> int:
    return sum(len(m.content.split()) for m in req.messages)
