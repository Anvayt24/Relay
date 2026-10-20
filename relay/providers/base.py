"""The provider adapter interface (PRD §9.3, FR-2).

An adapter translates between Relay's internal schema and one provider's API. It does no
retries: provider SDK retries are disabled, and Relay's reliability layer is the single
retry owner. It reports failures as ``ProviderError`` with an Appendix B ``ErrorClass``.

``stream()`` contract:
- **It always ends with a usage chunk**, whether or not the client asked for usage. Relay
  needs the token counts for logging now, and for cost and reservations later. It decides
  separately whether to forward the chunk.
- **It returns an ``AsyncGenerator``, not just an ``AsyncIterator``** (PRD §9.3 says
  ``AsyncIterator``), because Relay must be able to ``aclose()`` it. Closing the generator is
  how a client disconnect reaches the provider. A real adapter holds the upstream HTTP
  response open inside the generator (``async with client.stream(...)``), so ``aclose()``
  closes the connection, and the provider stops generating (and billing for) tokens nobody
  will read.
"""

from collections.abc import AsyncGenerator
from typing import Protocol

from relay.schemas.chat import ChatChunk, ChatRequest, ChatResponse


class ProviderAdapter(Protocol):
    name: str

    async def complete(self, req: ChatRequest) -> ChatResponse: ...

    def stream(self, req: ChatRequest) -> AsyncGenerator[ChatChunk, None]: ...

    # Releases pooled connections. The app's lifespan calls it at shutdown.
    async def aclose(self) -> None: ...
