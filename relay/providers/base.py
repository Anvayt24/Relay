"""The provider adapter interface (PRD §9.3, FR-2).

An adapter translates between Relay's internal schema and one provider's API. It does no
retries: provider SDK retries are disabled, and Relay's reliability layer is the single
retry owner. It reports failures as ``ProviderError`` with an Appendix B ``ErrorClass``.

Phase 1 has only ``complete()``. ``stream()`` is added in Phase 2, together with the code that
consumes it.
"""

from typing import Protocol

from relay.schemas.chat import ChatRequest, ChatResponse


class ProviderAdapter(Protocol):
    name: str

    async def complete(self, req: ChatRequest) -> ChatResponse: ...
