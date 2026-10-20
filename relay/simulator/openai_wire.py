"""OpenAI's wire format, as a *provider* speaks it (the simulator's side of the connection).

This is the mirror image of ``relay/api/``. There, Relay is the server and must *produce*
OpenAI-compatible responses for its clients. Here the simulator plays OpenAI (or Groq, or
OpenRouter, which copy the format), and what matters is reproducing the provider's
behavior, quirks included, so Phase 4's adapters are tested against something realistic:

- Errors use OpenAI's ``type``/``code`` pairs. The adapter must tell ``rate_limit_exceeded``
  from ``insufficient_quota`` (same 429, different meaning: Appendix B).
- 429s carry ``retry-after`` (whole seconds) and ``retry-after-ms``, plus ``x-ratelimit-*``
  headers on every response.
- A mid-stream failure is **not** an error event. A real provider's connection just drops.
  That case is handled in ``server.py``, because it is about the connection, not the format.
"""

import math
from typing import Any

from relay.schemas.chat import ChatChunk, ChatRequest, ChatResponse
from relay.schemas.errors import ErrorClass, ProviderError
from relay.simulator.capacity import LimitState

# class -> (status, type, code, message), modelled on OpenAI's documented errors.
ERRORS: dict[ErrorClass, tuple[int, str, str | None, str]] = {
    ErrorClass.RATE_LIMITED: (429, "requests", "rate_limit_exceeded", "Rate limit reached."),
    ErrorClass.QUOTA_EXHAUSTED: (
        429,
        "insufficient_quota",
        "insufficient_quota",
        "You exceeded your current quota, please check your plan and billing details.",
    ),
    ErrorClass.OVERLOADED: (503, "server_error", None, "The engine is currently overloaded."),
    ErrorClass.SERVER_ERROR: (500, "server_error", None, "The server had an error."),
    ErrorClass.TIMEOUT: (504, "server_error", None, "Gateway timeout."),
    ErrorClass.BAD_REQUEST: (400, "invalid_request_error", None, "Invalid request."),
    ErrorClass.CONTEXT_LENGTH_EXCEEDED: (
        400,
        "invalid_request_error",
        "context_length_exceeded",
        "This model's maximum context length was exceeded.",
    ),
    ErrorClass.MODEL_NOT_FOUND: (
        404,
        "invalid_request_error",
        "model_not_found",
        "The model does not exist or you do not have access to it.",
    ),
    ErrorClass.AUTH: (
        401,
        "invalid_request_error",
        "invalid_api_key",
        "Incorrect API key provided.",
    ),
    ErrorClass.CONTENT_FILTERED: (
        400,
        "invalid_request_error",
        "content_filter",
        "The response was filtered due to the prompt triggering the content management policy.",
    ),
}


def error(
    err: ProviderError, limits: LimitState | None
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """(status, body, headers) for a failure before the response started."""
    status, type_, code, message = ERRORS.get(err.error_class, ERRORS[ErrorClass.SERVER_ERROR])
    if err.error_class in (ErrorClass.BAD_REQUEST, ErrorClass.MODEL_NOT_FOUND):
        message = err.message  # these carry a specific reason worth showing
    body = {"error": {"message": message, "type": type_, "param": None, "code": code}}
    headers = rate_limit_headers(limits)
    if err.signal is not None and err.signal.retry_after_s is not None:
        headers["retry-after"] = str(math.ceil(err.signal.retry_after_s))
        headers["retry-after-ms"] = str(math.ceil(err.signal.retry_after_s * 1000))
    return status, body, headers


def rate_limit_headers(limits: LimitState | None) -> dict[str, str]:
    if limits is None:
        return {}
    return {
        "x-ratelimit-limit-requests": str(limits.limit_requests),
        "x-ratelimit-limit-tokens": str(limits.limit_tokens),
        "x-ratelimit-remaining-requests": str(limits.remaining_requests),
        "x-ratelimit-remaining-tokens": str(limits.remaining_tokens),
        "x-ratelimit-reset-requests": duration(limits.reset_requests_s),
        "x-ratelimit-reset-tokens": duration(limits.reset_tokens_s),
    }


def duration(seconds: float) -> str:
    """OpenAI's reset format: Go-style durations such as ``"120ms"``, ``"1.5s"``, ``"6m0s"``.

    An adapter has to parse this (Phase 4), so the simulator must produce it, not plain
    seconds.
    """
    if seconds < 1:
        return f"{math.ceil(seconds * 1000)}ms"
    minutes, secs = divmod(seconds, 60)
    secs_text = f"{secs:.3f}".rstrip("0").rstrip(".")
    return f"{int(minutes)}m{secs_text}s" if minutes else f"{secs_text}s"


def response_body(response: ChatResponse) -> dict[str, Any]:
    return response.model_dump(mode="json")


def include_usage(req: ChatRequest) -> bool:
    return bool(req.stream_options and req.stream_options.include_usage)


def chunk_body(chunk: ChatChunk) -> dict[str, Any]:
    return chunk.model_dump(mode="json", exclude_none=True)
