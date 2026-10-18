"""The Stage protocol and its result types (PRD §9.3).

A request is handled by an ordered list of stages, for example
log → auth → admission → cache → router → reservation → provider. Each stage either:

- passes the request on (``Continue``);
- answers it itself (``ShortCircuit``), e.g. a cache hit or the provider's response;
- refuses it (``Reject``), e.g. over budget or unknown model.

**Convention:** an *expected* failure is a ``Reject``, and an *exception* means a bug. That
keeps "the request was refused" (normal, part of the API) separate from "the code broke"
(alert someone).

Every stage that ran also gets ``finalize(ctx, outcome)``, in reverse order, once the request
is over. That is where cleanup that depends on the result goes: reconciling a reservation
against actual tokens, enqueueing the log row, sampling for evidence. ``process()`` cannot do
those jobs, because when it runs the result doesn't exist yet.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

from relay.pipeline.context import RequestContext
from relay.pipeline.streaming import PrimedStream
from relay.schemas.chat import ChatResponse
from relay.schemas.errors import ErrorClass, RelayError


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class ShortCircuit:
    response: ChatResponse | PrimedStream


@dataclass(frozen=True)
class Reject:
    error: RelayError


type StageResult = Continue | ShortCircuit | Reject


@dataclass(frozen=True)
class Outcome:
    """How the request ended, as seen by finalize().

    ``failed_midstream`` is separate from ``error`` because the client already received part
    of an answer. That difference matters for billing, reservations and evidence sampling.
    """

    status: Literal["success", "error", "cancelled", "failed_midstream"]
    error_class: ErrorClass | None = None


class Stage(Protocol):
    """Structural interface: any class with these members is a Stage, and mypy checks it.

    Protocol rather than an ABC because stages (and the fakes in tests) need no shared base
    class or implementation. They only need the right shape. An ABC would add an inheritance
    requirement and buy nothing.
    """

    name: str

    async def process(self, ctx: RequestContext) -> StageResult: ...

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None: ...
