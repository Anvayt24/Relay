"""Priming a stream: deciding *when* Relay commits to it (PRD FR-5, "first byte").

HTTP sends the status line and headers before any body bytes. Once ``200 OK`` is out, the
client can't be told "actually that failed, here's a 503". So the moment Relay sends its first
body byte, it is committed to this upstream.

``prime()`` therefore pulls upstream chunks *without sending anything* until the first chunk
that carries text, and buffers the role-only chunk that comes before it. Until then:
- a failure is still an ordinary HTTP error;
- Phase 7 can transparently try another provider;
- the ``X-Relay-Provider``/``X-Relay-Model`` headers, sent with the commit, name the provider
  that actually answered.

Waiting for the role chunk alone would commit before knowing whether the provider can produce
any text at all.
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from relay.schemas.chat import ChatChunk
from relay.schemas.errors import ErrorClass, ProviderError


@dataclass
class PrimedStream:
    buffered: list[ChatChunk]  # everything up to and including the first content chunk
    upstream: AsyncGenerator[ChatChunk, None]  # positioned just after `buffered`
    relay_model: str

    def relabel(self, chunk: ChatChunk) -> ChatChunk:
        # Clients see Relay names, never upstream IDs, in streams as in JSON responses.
        return chunk.model_copy(update={"model": self.relay_model})


async def prime(
    upstream: AsyncGenerator[ChatChunk, None], *, timeout_s: float, relay_model: str
) -> PrimedStream:
    primed = PrimedStream(buffered=[], upstream=upstream, relay_model=relay_model)
    committed = False
    try:
        async with asyncio.timeout(timeout_s):
            # `break` leaves an async generator open (unlike exhausting it), which is exactly
            # what is wanted: the rest is consumed later, by the response body.
            async for chunk in upstream:
                primed.buffered.append(primed.relabel(chunk))
                # Also stop at a finish_reason. An empty completion has no content chunk, and
                # waiting for one would hang until the timeout.
                if chunk.has_content or chunk.is_finished:
                    break
        committed = True
        return primed
    except TimeoutError:
        raise ProviderError(
            ErrorClass.TIMEOUT, f"no content from upstream within {timeout_s}s"
        ) from None
    finally:
        if not committed:
            # Not handing the stream on, so close it here. This releases the upstream
            # connection now, not whenever the garbage collector gets to it.
            await upstream.aclose()
