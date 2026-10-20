"""Adapter for Anthropic's Messages API (PRD FR-2: "expect most adapter bugs there").

The translation, both ways:

    Relay (OpenAI-shaped)              Anthropic
    system messages              ->    top-level "system" string
    max_tokens (optional)        ->    max_tokens (required; Relay always fills it, see
                                       ProviderStage and ModelSpec.default_max_tokens)
    stop                         ->    stop_sequences
    finish_reason stop/length    <-    stop_reason end_turn|stop_sequence / max_tokens
    usage                        <-    input_tokens in message_start, output_tokens in
                                       message_delta (split!), plus cache read/write counts

Stream events become Relay chunks:
- ``message_start`` → the role chunk;
- ``content_block_delta`` with ``text_delta`` → a content chunk (other block types, such as
  thinking or tool input, are skipped: tools are unsupported at P0);
- ``message_delta`` → the finish chunk;
- ``message_stop`` → the end, then Relay's usage chunk;
- ``ping`` → ignored;
- ``error`` → ``ProviderError``.

A stream that ends without ``message_stop`` was cut off.
"""

import json
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Literal

import httpx

from relay.config.models import LimitScope
from relay.providers.http import (
    error_payload,
    int_header,
    retry_after,
    sse_events,
    transport_error,
)
from relay.schemas.chat import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    Choice,
    ChunkChoice,
    ChunkDelta,
    PromptTokensDetails,
    ResponseMessage,
    Usage,
)
from relay.schemas.errors import ErrorClass, ProviderError, ProviderSignal

API_VERSION = "2023-06-01"  # the Messages API version header Anthropic requires
FinishReason = Literal["stop", "length"]


class AnthropicAdapter:
    def __init__(
        self,
        name: str = "anthropic",
        *,
        base_url: str,
        api_key: str,
        limit_scope: LimitScope = "model",
        default_max_tokens: int = 1024,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 60.0,
        stream_idle_timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self._limit_scope: LimitScope = limit_scope
        self._default_max_tokens = default_max_tokens
        self._stream_timeout = httpx.Timeout(
            connect=connect_timeout_s,
            read=stream_idle_timeout_s,
            write=10.0,
            pool=connect_timeout_s,
        )
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"x-api-key": api_key, "anthropic-version": API_VERSION},
            timeout=httpx.Timeout(
                connect=connect_timeout_s, read=read_timeout_s, write=10.0, pool=connect_timeout_s
            ),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ complete
    async def complete(self, req: ChatRequest) -> ChatResponse:
        try:
            response = await self._client.post("/v1/messages", json=self._body(req))
        except httpx.HTTPError as exc:
            raise transport_error(exc, content_started=False) from exc
        if response.status_code >= 400:
            raise self._error(response.status_code, response.content, response.headers)
        try:
            data = response.json()
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            usage = _usage(data["usage"], data["usage"].get("output_tokens", 0))
            return ChatResponse(
                id=str(data.get("id", "msg_upstream")),
                created=int(time.time()),
                model=str(data.get("model") or req.model),
                choices=[
                    Choice(
                        message=ResponseMessage(content=text),
                        finish_reason=_finish(data.get("stop_reason")),
                    )
                ],
                usage=usage,
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(ErrorClass.SERVER_ERROR, f"malformed response: {exc!r}") from exc

    # ------------------------------------------------------------------ stream
    async def stream(self, req: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        body = {**self._body(req), "stream": True}
        msg_id, model, created = "msg_upstream", req.model, int(time.time())
        input_usage: dict[str, Any] = {}
        output_tokens = 0
        content_seen = stopped = False

        def chunk(delta: ChunkDelta, finish: FinishReason | None = None) -> ChatChunk:
            return ChatChunk(
                id=msg_id,
                created=created,
                model=model,
                choices=[ChunkChoice(delta=delta, finish_reason=finish)],
            )

        try:
            async with self._client.stream(
                "POST", "/v1/messages", json=body, timeout=self._stream_timeout
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise self._error(response.status_code, response.content, response.headers)
                async for event in sse_events(response.aiter_lines()):
                    try:
                        data = json.loads(event.data)
                    except ValueError:
                        raise ProviderError(
                            ErrorClass.STREAM_BROKEN if content_seen else ErrorClass.SERVER_ERROR,
                            f"malformed event: {event.data[:100]!r}",
                        ) from None
                    kind = event.event or data.get("type")

                    if kind == "message_start":
                        message = data.get("message", {})
                        msg_id = str(message.get("id", msg_id))
                        model = str(message.get("model", model))
                        input_usage = message.get("usage", {})
                        yield chunk(ChunkDelta(role="assistant", content=""))
                    elif kind == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            content_seen = True
                            yield chunk(ChunkDelta(content=delta["text"]))
                    elif kind == "message_delta":
                        output_tokens = int(data.get("usage", {}).get("output_tokens", 0))
                        stop_reason = data.get("delta", {}).get("stop_reason")
                        yield chunk(ChunkDelta(), _finish(stop_reason))
                    elif kind == "message_stop":
                        stopped = True
                        break
                    elif kind == "error":
                        # In-band failure after the 200 (Anthropic's way: OpenAI's is a dropped
                        # connection). After content it can only be stream_broken.
                        error, blob = error_payload(event.data.encode())
                        if content_seen:
                            raise ProviderError(ErrorClass.STREAM_BROKEN, f"error event: {blob}")
                        raise ProviderError(
                            classify(_status_for_type(str(error.get("type", ""))), blob),
                            f"error event: {blob}",
                        )
                    # ping, content_block_start/stop: nothing to forward.
        except httpx.HTTPError as exc:
            raise transport_error(exc, content_started=content_seen) from exc

        if not stopped:
            raise ProviderError(
                ErrorClass.STREAM_BROKEN if content_seen else ErrorClass.SERVER_ERROR,
                "stream ended before message_stop",
            )
        yield ChatChunk(
            id=msg_id,
            created=created,
            model=model,
            choices=[],
            usage=_usage(input_usage, output_tokens),
        )

    # ------------------------------------------------------------------ helpers
    def _body(self, req: ChatRequest) -> dict[str, Any]:
        system = "\n\n".join(m.content for m in req.messages if m.role == "system")
        body: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens or self._default_max_tokens,
            "messages": [
                {"role": m.role, "content": m.content} for m in req.messages if m.role != "system"
            ],
        }
        optional = {
            "system": system or None,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stop_sequences": req.stop,
        }
        body.update({k: v for k, v in optional.items() if v is not None})
        return body

    def _error(self, status: int, body: bytes, headers: Mapping[str, str]) -> ProviderError:
        error, blob = error_payload(body)
        error_class = classify(status, blob)
        signal = None
        if error_class in (ErrorClass.RATE_LIMITED, ErrorClass.OVERLOADED):
            signal = ProviderSignal(
                retry_after_s=retry_after(headers),
                limit_scope=self._limit_scope,
                remaining_requests=int_header(headers, "anthropic-ratelimit-requests-remaining"),
                remaining_tokens=int_header(headers, "anthropic-ratelimit-tokens-remaining"),
            )
        message = f"{self.name} {status}: {str(error.get('message', ''))[:300]}"
        return ProviderError(error_class, message, signal)


def classify(status: int, blob: str) -> ErrorClass:
    if status == 529 or "overloaded" in blob:
        return ErrorClass.OVERLOADED
    if status == 429:
        return ErrorClass.RATE_LIMITED
    if status == 504 or "timeout_error" in blob:
        return ErrorClass.TIMEOUT
    if status >= 500:
        return ErrorClass.SERVER_ERROR
    # Out of credit is a 400, recognizable only by its message.
    if "credit balance" in blob:
        return ErrorClass.QUOTA_EXHAUSTED
    if "prompt is too long" in blob or "too many tokens" in blob:
        return ErrorClass.CONTEXT_LENGTH_EXCEEDED
    if "content filtering" in blob or "output blocked" in blob:
        return ErrorClass.CONTENT_FILTERED
    if status in (401, 403):
        return ErrorClass.AUTH
    if status == 404:
        return ErrorClass.MODEL_NOT_FOUND
    return ErrorClass.BAD_REQUEST


def _status_for_type(error_type: str) -> int:
    # An error *event* has no HTTP status; recover the equivalent from its type.
    return {
        "overloaded_error": 529,
        "rate_limit_error": 429,
        "api_error": 500,
        "timeout_error": 504,
        "authentication_error": 401,
        "permission_error": 403,
        "not_found_error": 404,
    }.get(error_type, 400)


def _finish(stop_reason: object) -> FinishReason:
    return "length" if stop_reason == "max_tokens" else "stop"


def _usage(input_usage: Mapping[str, Any], output_tokens: int) -> Usage:
    # Anthropic counts cache reads and writes *separately* from input_tokens; OpenAI's
    # prompt_tokens includes them. Sum them so prompt_tokens means the same everywhere.
    uncached = int(input_usage.get("input_tokens", 0))
    cache_read = int(input_usage.get("cache_read_input_tokens") or 0)
    cache_write = int(input_usage.get("cache_creation_input_tokens") or 0)
    prompt = uncached + cache_read + cache_write
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=output_tokens,
        total_tokens=prompt + output_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cache_read) if cache_read else None,
        cache_write_tokens=cache_write,
    )
