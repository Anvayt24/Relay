"""Anthropic's Messages API wire format, as the provider speaks it.

The same engine serves this format; only the translation differs. These are the differences
Phase 4's Anthropic adapter must handle, and why the simulator reproduces them:

- **``system`` is a top-level field**, not a message. ``max_tokens`` is **required**.
- **Usage is split**: input tokens arrive in ``message_start`` at the beginning of a stream,
  output tokens in ``message_delta`` at the end. An adapter that reads only one event
  under-reports cost.
- **Stream events are named**, using the SSE ``event:`` field:
  ``message_start → content_block_start → ping → content_block_delta* → content_block_stop →
  message_delta → message_stop``. There is no ``[DONE]``; ``message_stop`` ends a stream.
- **Overload is HTTP 529**, a non-standard status. A client that only knows the standard
  5xx codes misclassifies it.
- **A mid-stream failure is an ``event: error`` frame** on a stream that already returned
  200, not a dropped connection as with OpenAI.
- **Running out of credit is a 400**, not a 429, recognizable only by its message. This is the
  kind of quirk Appendix B's "adapters distinguish rate_limited from quota_exhausted using
  each provider's headers and error body" is about.

Error bodies and messages are modelled on Anthropic's documented error types.
"""

import json
import math
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from relay.schemas.chat import ChatChunk, ChatRequest, ChatResponse
from relay.schemas.errors import ErrorClass, ProviderError
from relay.simulator.capacity import LimitState
from relay.simulator.engine import Call

STOP_REASON = {"stop": "end_turn", "length": "max_tokens"}

# class -> (status, Anthropic error type, message)
ERRORS: dict[ErrorClass, tuple[int, str, str]] = {
    ErrorClass.RATE_LIMITED: (
        429,
        "rate_limit_error",
        "Number of requests has exceeded your rate limit.",
    ),
    ErrorClass.QUOTA_EXHAUSTED: (
        400,
        "invalid_request_error",
        "Your credit balance is too low to access the Anthropic API.",
    ),
    ErrorClass.OVERLOADED: (529, "overloaded_error", "Overloaded"),
    ErrorClass.SERVER_ERROR: (500, "api_error", "Internal server error"),
    ErrorClass.TIMEOUT: (504, "timeout_error", "Request timed out"),
    ErrorClass.BAD_REQUEST: (400, "invalid_request_error", "Invalid request"),
    ErrorClass.CONTEXT_LENGTH_EXCEEDED: (400, "invalid_request_error", "prompt is too long"),
    ErrorClass.MODEL_NOT_FOUND: (404, "not_found_error", "model not found"),
    ErrorClass.AUTH: (401, "authentication_error", "invalid x-api-key"),
    ErrorClass.CONTENT_FILTERED: (
        400,
        "invalid_request_error",
        "Output blocked by content filtering policy",
    ),
    # Only seen mid-stream, as an error event: an overload is the usual real-world cause.
    ErrorClass.STREAM_BROKEN: (529, "overloaded_error", "Overloaded"),
}


class TextBlock(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[TextBlock]


class MessagesRequest(BaseModel):
    # Sampling fields (temperature, top_p, stop_sequences...) are accepted and ignored: they
    # do not change what the simulator does.
    model_config = ConfigDict(extra="allow")

    model: str
    max_tokens: int = Field(ge=1)
    messages: list[Message] = Field(min_length=1)
    system: str | list[TextBlock] | None = None
    stream: bool = False


def _text(content: str | list[TextBlock]) -> str:
    return content if isinstance(content, str) else "\n".join(b.text for b in content)


def to_chat_request(body: object) -> ChatRequest:
    """Translate into the engine's request. Raises pydantic.ValidationError on a bad body."""
    req = MessagesRequest.model_validate(body)
    messages = [{"role": m.role, "content": _text(m.content)} for m in req.messages]
    if req.system is not None:
        messages.insert(0, {"role": "system", "content": _text(req.system)})
    return ChatRequest.model_validate(
        {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "stream": req.stream,
        }
    )


def message_id(call: Call) -> str:
    return call.id.replace("chatcmpl-", "msg_", 1)


def response_body(call: Call, response: ChatResponse) -> dict[str, Any]:
    choice = response.choices[0]
    return {
        "id": message_id(call),
        "type": "message",
        "role": "assistant",
        "model": response.model,
        "content": [{"type": "text", "text": choice.message.content}],
        "stop_reason": STOP_REASON[choice.finish_reason],
        "stop_sequence": None,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    }


def error(
    err: ProviderError, limits: LimitState | None
) -> tuple[int, dict[str, Any], dict[str, str]]:
    status, _, message = ERRORS.get(err.error_class, ERRORS[ErrorClass.SERVER_ERROR])
    if err.error_class in (ErrorClass.BAD_REQUEST, ErrorClass.MODEL_NOT_FOUND):
        message = err.message
    headers = rate_limit_headers(limits)
    if err.signal is not None and err.signal.retry_after_s is not None:
        headers["retry-after"] = str(math.ceil(err.signal.retry_after_s))
    return status, error_body(err.error_class, message), headers


def error_body(error_class: ErrorClass, message: str | None = None) -> dict[str, Any]:
    _, type_, default = ERRORS.get(error_class, ERRORS[ErrorClass.SERVER_ERROR])
    return {"type": "error", "error": {"type": type_, "message": message or default}}


def rate_limit_headers(limits: LimitState | None) -> dict[str, str]:
    if limits is None:
        return {}
    # Anthropic reports resets as RFC 3339 timestamps, not durations (OpenAI's style).
    now = datetime.now(UTC)

    def at(seconds: float) -> str:
        return (
            (now + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace("+00:00", "Z")
        )

    return {
        "anthropic-ratelimit-requests-limit": str(limits.limit_requests),
        "anthropic-ratelimit-requests-remaining": str(limits.remaining_requests),
        "anthropic-ratelimit-requests-reset": at(limits.reset_requests_s),
        "anthropic-ratelimit-tokens-limit": str(limits.limit_tokens),
        "anthropic-ratelimit-tokens-remaining": str(limits.remaining_tokens),
        "anthropic-ratelimit-tokens-reset": at(limits.reset_tokens_s),
    }


def _event(name: str, data: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


async def events(call: Call, rest: AsyncGenerator[ChatChunk, None]) -> AsyncGenerator[bytes, None]:
    """Re-shape the engine's chunks (after the role chunk) into Anthropic's events."""
    yield _event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id(call),
                "type": "message",
                "role": "assistant",
                "model": call.req.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": call.prompt_tokens, "output_tokens": 0},
            },
        },
    )
    yield _event(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )
    yield _event("ping", {"type": "ping"})

    stop_reason = "end_turn"
    try:
        async for chunk in rest:
            if chunk.usage is not None:
                yield _event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": chunk.usage.completion_tokens},
                    },
                )
                yield _event("message_stop", {"type": "message_stop"})
            elif chunk.has_content:
                text = chunk.choices[0].delta.content or ""
                yield _event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            elif chunk.is_finished:
                stop_reason = STOP_REASON[chunk.choices[0].finish_reason or "stop"]
                yield _event("content_block_stop", {"type": "content_block_stop", "index": 0})
    except ProviderError as err:
        # Anthropic reports a failure after the 200 as an error *event*, then ends the stream.
        yield _event("error", error_body(err.error_class))
