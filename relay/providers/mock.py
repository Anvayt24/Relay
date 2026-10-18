"""A deterministic in-process provider for tests, local runs and demos.

Given the same request it always gives the same reply, with configurable failures. That makes
gateway behavior testable without network, cost or flakiness. Phase 3 adds a richer simulator
(latency distributions, error rates, ground-truth quality) in ``bench/simulator``.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from relay.schemas.chat import ChatRequest, ChatResponse, Choice, ResponseMessage, Usage
from relay.schemas.errors import ErrorClass, ProviderError, ProviderSignal


@dataclass(frozen=True)
class MockBehavior:
    latency_s: float = 0.0
    error: ErrorClass | None = None  # raise this instead of answering
    retry_after_s: float | None = None  # sent with the error, as a real 429 would
    reply_prefix: str = "mock: "


class MockProvider:
    name = "mock"

    def __init__(self, behavior: MockBehavior | None = None) -> None:
        self.behavior = behavior or MockBehavior()
        # How many times complete() was called. Tests use it to prove nobody retried.
        self.calls = 0

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.behavior.latency_s > 0:
            # asyncio.sleep, never time.sleep: a blocking sleep would freeze the event loop,
            # and with it every other request in this process.
            await asyncio.sleep(self.behavior.latency_s)

        if self.behavior.error is not None:
            signal = None
            if self.behavior.retry_after_s is not None:
                signal = ProviderSignal(
                    retry_after_s=self.behavior.retry_after_s, limit_scope="model"
                )
            raise ProviderError(self.behavior.error, f"mock injected {self.behavior.error}", signal)

        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        # "Tokens" are whitespace-separated words. That is enough to exercise usage accounting
        # and max_tokens; real tokenizers arrive with admission estimates (Phase 6).
        words = f"{self.behavior.reply_prefix}{last_user}".split()
        finish_reason: Literal["stop", "length"] = "stop"
        if req.max_tokens is not None and len(words) > req.max_tokens:
            words = words[: req.max_tokens]
            finish_reason = "length"

        prompt_tokens = sum(len(m.content.split()) for m in req.messages)
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
