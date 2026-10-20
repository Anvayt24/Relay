"""The streaming HTTP response: SSE body, mid-stream policy, and stream cleanup.

By the time this response exists, the pipeline has already primed the stream (the first
content chunk is buffered), so committing ``200 OK`` and the headers is safe. From here on,
FR-5's mid-stream policy applies:

- **Success:** the chunks, then (only if the client asked) the usage chunk, then ``[DONE]``.
- **Failure after the commit** (provider error or idle timeout): an SSE error payload,
  *no* ``[DONE]``, and the outcome ``failed_midstream``. There is no switch to another model:
  the client would get the first half of one model's answer glued to another's.
- **The client goes away:** the stream is closed and the outcome is ``cancelled``.

**Who cleans up.** Starlette's ``StreamingResponse`` never closes its body iterator.
- When a send fails, it raises ``ClientDisconnect``.
- When it sees ``http.disconnect`` (uvicorn uses ASGI spec 2.3), it *cancels* the task that
  is streaming.

Either way, the async generators are left suspended, and their ``finally`` blocks would run
whenever the garbage collector happens to finalize them, which is too late to close an
upstream connection or write the log row.

So this class overrides ``__call__``. The ASGI server always awaits it to completion or
cancellation, which makes its ``finally`` the one place guaranteed to run. That ``finally``
closes the body, closes the upstream, and calls the pipeline's ``Finalizer``, which this
response took ownership of (see ``relay/pipeline/runner.py``).
"""

import asyncio
import time
from collections.abc import AsyncGenerator, Mapping

from prometheus_client import Counter
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from relay.admission.tokens import text_tokens
from relay.api.sse import DONE, encode_chunk, encode_error
from relay.config.pricing import usd
from relay.pipeline.context import RequestContext
from relay.pipeline.runner import StreamResult
from relay.pipeline.stage import Outcome
from relay.schemas.chat import ChatChunk
from relay.schemas.errors import ErrorClass, ProviderError

FAILED_MIDSTREAM = Counter(
    "relay_stream_failed_midstream_total",
    "Streams that broke after the first content chunk was sent.",
    ["provider"],
)

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    # Reverse proxies such as nginx buffer responses by default, which would turn a stream
    # into one late blob. This header tells them not to.
    "X-Accel-Buffering": "no",
}


class RelayStreamingResponse(StreamingResponse):
    def __init__(
        self,
        result: StreamResult,
        ctx: RequestContext,
        *,
        idle_timeout_s: float,
        headers: Mapping[str, str],
    ) -> None:
        self._result = result
        self._ctx = ctx
        self._idle_timeout_s = idle_timeout_s
        self._outcome: Outcome | None = None  # set by the body; None at the end = cancelled
        self._body = self._events()
        super().__init__(
            self._body,
            media_type="text/event-stream",
            headers={**STREAM_HEADERS, **headers},
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Order matters: stop producing (body, then upstream), then record what happened.
            # aclose() is safe on a generator that already finished, or never started.
            await self._body.aclose()
            await self._result.stream.upstream.aclose()
            await self._result.finalize(self._outcome or Outcome("cancelled"))

    async def _events(self) -> AsyncGenerator[bytes, None]:
        stream = self._result.stream
        request = self._ctx.request
        include_usage = bool(request.stream_options and request.stream_options.include_usage)

        try:
            for chunk in stream.buffered:
                if chunk.has_content and self._ctx.ttft_s is None:
                    self._ctx.ttft_s = time.monotonic() - self._ctx.started_at
                self._count_forwarded(chunk)
                yield encode_chunk(chunk)

            while True:
                try:
                    # The idle timeout is per chunk, not per stream. A long answer that keeps
                    # flowing is fine; an upstream that goes quiet for 30 s is not. The
                    # timeout cancels the upstream's pending await, so the upstream generator
                    # sees CancelledError at the point where it was waiting.
                    async with asyncio.timeout(self._idle_timeout_s):
                        chunk = await anext(stream.upstream)
                except StopAsyncIteration:
                    break
                chunk = stream.relabel(chunk)
                if chunk.usage is not None:
                    self._ctx.record_usage(chunk.usage)  # always: for logs and cost
                    if include_usage:
                        yield encode_chunk(chunk, extra={"relay": self._relay_extension()})
                    continue
                self._count_forwarded(chunk)
                yield encode_chunk(chunk)
        except (ProviderError, TimeoutError) as e:
            self._outcome = Outcome("failed_midstream", ErrorClass.STREAM_BROKEN)
            provider = self._ctx.attempts[-1].provider if self._ctx.attempts else "unknown"
            FAILED_MIDSTREAM.labels(provider=provider).inc()
            reason = "upstream went idle" if isinstance(e, TimeoutError) else str(e)
            yield encode_error(ErrorClass.STREAM_BROKEN, f"stream broken: {reason}")
            return  # no [DONE]: the answer is incomplete, and the client must not think otherwise

        self._outcome = Outcome("success")
        yield DONE

    def _count_forwarded(self, chunk: ChatChunk) -> None:
        # Counted *before* the yield: once yielded, the client may already have it, and a
        # cancellation can arrive at the yield itself.
        for choice in chunk.choices:
            if choice.delta.content:
                self._ctx.forwarded_tokens += text_tokens(choice.delta.content)

    def _relay_extension(self) -> dict[str, str]:
        # Extra top-level key on the final usage chunk (FR-1). OpenAI SDKs ignore unknown
        # fields. The cost lives here because a header can't carry it: headers were sent with
        # the first chunk, before the usage (and so the cost) was known.
        attempt = self._ctx.attempts[-1]
        extension = {
            "request_id": self._ctx.request_id,
            "route": self._ctx.route,
            "provider": attempt.provider,
            "model": attempt.model,
        }
        if self._ctx.cost is not None:
            extension["cost_usd"] = usd(self._ctx.cost.billed_nano)
            extension["list_cost_usd"] = usd(self._ctx.cost.list_nano)
        return extension
