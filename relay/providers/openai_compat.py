"""Adapter for providers that speak OpenAI's chat-completions format: Groq, OpenRouter, and
Gemini's OpenAI-compatible endpoint (PRD FR-2).

"Compatible" is never exact. The differences this adapter absorbs, without per-provider
branches, by parsing tolerantly:
- **Usage location.** Usage arrives in a final chunk with ``choices: []`` when asked for with
  ``stream_options.include_usage``. Groq has also sent it as ``x_groq.usage``. If none arrives,
  a usage chunk is synthesized and ``relay_provider_usage_missing_total`` counts it, because
  the adapter contract promises a usage chunk (``providers/base.py``).
- **Errors inside a 200 stream.** OpenRouter reports a failure after the stream started as
  ``data: {"error": {...}}``.
- **Error body shapes.** These are a dict, a list (Gemini), or text from a proxy; see
  ``http.error_payload``.
- **Keep-alive comments.** OpenRouter sends ``: OPENROUTER PROCESSING`` lines.

How a stream's end is judged: it completed if a ``finish_reason`` arrived. ``[DONE]`` is the
usual terminator, but treating "finish seen" as the criterion tolerates a provider that omits
``[DONE]``, while still catching the real failure, a stream cut off mid-answer.
"""

import json
import logging
import re
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any, Literal

import httpx
from prometheus_client import Counter

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

logger = logging.getLogger(__name__)

USAGE_MISSING = Counter(
    "relay_provider_usage_missing_total",
    "Streams that ended without a usage report; usage was estimated instead.",
    ["provider"],
)

FinishReason = Literal["stop", "length"]
# (error event payload, whether content was already seen) -> the error to raise
ErrorFromEvent = Callable[[dict[str, Any], bool], ProviderError]

# A 429 can mean "slow down for a few seconds" or "you are out for the day": very different
# reactions (Appendix B). These words in the error mark the second kind.
_QUOTA = re.compile(r"insufficient_quota|per[-_ ]?day|daily|credits?")
_CONTEXT = re.compile(r"context[_ ]length|maximum context|context window|too many tokens|too long")
_FILTERED = re.compile(r"content[_ ]filter|content management|moderation|flagged|safety")


class OpenAICompatAdapter:
    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        api_key: str,
        limit_scope: LimitScope = "model",
        max_tokens_param: str = "max_tokens",
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 60.0,
        stream_idle_timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self._limit_scope: LimitScope = limit_scope
        self._max_tokens_param = max_tokens_param
        self._stream_timeout = httpx.Timeout(
            connect=connect_timeout_s,
            read=stream_idle_timeout_s,
            write=10.0,
            pool=connect_timeout_s,
        )
        # One pooled client per adapter, reused by every request: TCP and TLS setup happen
        # once per connection, not once per call. Built here (no I/O), closed in aclose().
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            # Non-streaming: the provider sends nothing until the whole answer is ready, so
            # the read timeout must cover the full generation.
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
            response = await self._client.post("/chat/completions", json=self._body(req))
        except httpx.HTTPError as exc:
            raise transport_error(exc, content_started=False) from exc
        if response.status_code >= 400:
            raise self._error(response.status_code, response.content, response.headers)
        try:
            return self._to_response(response.json(), req)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise ProviderError(ErrorClass.SERVER_ERROR, f"malformed response: {exc!r}") from exc

    def _to_response(self, data: dict[str, Any], req: ChatRequest) -> ChatResponse:
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content") or ""
        raw_finish = choice.get("finish_reason")
        if raw_finish == "content_filter" and not content:
            raise ProviderError(ErrorClass.CONTENT_FILTERED, "the provider filtered the response")
        usage = _usage(data.get("usage")) or _estimated_usage(req, completion=len(content.split()))
        return ChatResponse(
            id=str(data.get("id") or "chatcmpl-upstream"),
            created=int(data.get("created") or time.time()),
            model=str(data.get("model") or req.model),
            choices=[
                Choice(message=ResponseMessage(content=content), finish_reason=_finish(raw_finish))
            ],
            usage=usage,
        )

    # ------------------------------------------------------------------ stream
    async def stream(self, req: ChatRequest) -> AsyncGenerator[ChatChunk, None]:
        body = {**self._body(req), "stream": True, "stream_options": {"include_usage": True}}
        state = _StreamState(req)
        try:
            # `async with` keeps the HTTP response open exactly as long as this generator
            # lives. When Relay calls aclose() (client gone, timeout), GeneratorExit unwinds
            # through here, the response is closed, and the provider stops generating.
            async with self._client.stream(
                "POST", "/chat/completions", json=body, timeout=self._stream_timeout
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise self._error(response.status_code, response.content, response.headers)
                async for event in sse_events(response.aiter_lines()):
                    if event.data == "[DONE]":
                        break
                    for chunk in state.consume(event.data, self._error_from_event):
                        yield chunk
        except httpx.HTTPError as exc:
            raise transport_error(exc, content_started=state.content_seen) from exc

        if not state.finished:
            raise ProviderError(
                ErrorClass.STREAM_BROKEN if state.content_seen else ErrorClass.SERVER_ERROR,
                "stream ended before a finish_reason",
            )
        if state.usage is None:
            USAGE_MISSING.labels(provider=self.name).inc()
        yield state.usage_chunk()

    # ------------------------------------------------------------------ helpers
    def _body(self, req: ChatRequest) -> dict[str, Any]:
        # Built from named fields only: unknown client fields (ChatRequest keeps them, to
        # count them) must never be forwarded upstream.
        body: dict[str, Any] = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        }
        optional = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            self._max_tokens_param: req.max_tokens,
            "stop": req.stop,
            "seed": req.seed,
            "user": req.user,
        }
        body.update({k: v for k, v in optional.items() if v is not None})
        return body

    def _error(self, status: int, body: bytes, headers: Mapping[str, str]) -> ProviderError:
        error, blob = error_payload(body)
        message = f"{self.name} {status}: {str(error.get('message', ''))[:300]}"
        error_class = classify(status, blob)
        signal = None
        if error_class in (
            ErrorClass.RATE_LIMITED,
            ErrorClass.QUOTA_EXHAUSTED,
            ErrorClass.OVERLOADED,
        ):
            signal = ProviderSignal(
                retry_after_s=retry_after(headers),
                limit_scope=self._limit_scope,
                remaining_requests=int_header(headers, "x-ratelimit-remaining-requests"),
                remaining_tokens=int_header(headers, "x-ratelimit-remaining-tokens"),
            )
        return ProviderError(error_class, message, signal)

    def _error_from_event(self, payload: dict[str, Any], content_seen: bool) -> ProviderError:
        # An error *inside* a 200 stream (OpenRouter). After content, it can only be
        # stream_broken (Appendix B); before, classify it like any error.
        raw = json.dumps(payload).encode()
        _, blob = error_payload(raw)
        if content_seen:
            return ProviderError(ErrorClass.STREAM_BROKEN, f"{self.name} error event mid-stream")
        code = (
            payload.get("error", {}).get("code") if isinstance(payload.get("error"), dict) else None
        )
        status = code if isinstance(code, int) else 500
        return ProviderError(classify(status, blob), f"{self.name} error event before content")


def classify(status: int, blob: str) -> ErrorClass:
    """HTTP status + error text → Appendix B class. Order matters: specific before general."""
    if status == 402 or (status == 429 and _QUOTA.search(blob)):
        return ErrorClass.QUOTA_EXHAUSTED
    if status == 429:
        return ErrorClass.RATE_LIMITED
    if status in (503, 529):
        return ErrorClass.OVERLOADED
    if status in (408, 504):
        return ErrorClass.TIMEOUT
    if status >= 500:
        return ErrorClass.SERVER_ERROR
    if _FILTERED.search(blob):
        return ErrorClass.CONTENT_FILTERED
    if status in (400, 413, 422) and _CONTEXT.search(blob):
        return ErrorClass.CONTEXT_LENGTH_EXCEEDED
    if status in (401, 403):
        return ErrorClass.AUTH
    if status == 404:
        return ErrorClass.MODEL_NOT_FOUND
    return ErrorClass.BAD_REQUEST


class _StreamState:
    """What a stream has shown so far: turns raw ``data:`` payloads into ``ChatChunk``s."""

    def __init__(self, req: ChatRequest) -> None:
        self.req = req
        self.id = "chatcmpl-upstream"
        self.created = int(time.time())
        self.model = req.model
        self.content_seen = False
        self.finished = False
        self.content_chunks = 0
        self.usage: Usage | None = None

    def consume(self, data: str, on_error: ErrorFromEvent) -> list[ChatChunk]:
        try:
            payload = json.loads(data)
        except ValueError:
            raise ProviderError(
                ErrorClass.STREAM_BROKEN if self.content_seen else ErrorClass.SERVER_ERROR,
                f"malformed stream chunk: {data[:100]!r}",
            ) from None
        if not isinstance(payload, dict):
            return []
        if "error" in payload:
            raise on_error(payload, self.content_seen)
        self.id = str(payload.get("id") or self.id)
        self.model = str(payload.get("model") or self.model)
        self.created = int(payload.get("created") or self.created)

        usage = payload.get("usage") or (payload.get("x_groq") or {}).get("usage")
        if usage:
            self.usage = _usage(usage) or self.usage

        chunks = []
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            role = delta.get("role")
            raw_finish = choice.get("finish_reason")
            if content:
                self.content_seen = True
                self.content_chunks += 1
            if role or content:
                chunks.append(
                    self._chunk(ChunkDelta(role="assistant" if role else None, content=content))
                )
            if raw_finish:
                self.finished = True
                chunks.append(self._chunk(ChunkDelta(), _finish(raw_finish)))
        return chunks

    def usage_chunk(self) -> ChatChunk:
        usage = self.usage or _estimated_usage(self.req, completion=self.content_chunks)
        return ChatChunk(
            id=self.id, created=self.created, model=self.model, choices=[], usage=usage
        )

    def _chunk(self, delta: ChunkDelta, finish: FinishReason | None = None) -> ChatChunk:
        return ChatChunk(
            id=self.id,
            created=self.created,
            model=self.model,
            choices=[ChunkChoice(delta=delta, finish_reason=finish)],
        )


def _finish(raw: object) -> FinishReason:
    # OpenAI's enum also has tool_calls/function_call/content_filter. Relay's schema accepts
    # stop|length (tools are unsupported at P0), and anything else ended the answer normally.
    return "length" if raw == "length" else "stop"


def _usage(raw: object) -> Usage | None:
    if not isinstance(raw, dict) or "prompt_tokens" not in raw:
        return None
    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    details = raw.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=int(raw.get("total_tokens") or prompt + completion),
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached) if cached else None,
    )


def _estimated_usage(req: ChatRequest, *, completion: int) -> Usage:
    # Only when the provider reported nothing. Words stand in for tokens; Phase 6's estimator
    # is used for budgets, and the missing-usage metric says how often this happens.
    prompt = sum(len(m.content.split()) for m in req.messages)
    return Usage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )
