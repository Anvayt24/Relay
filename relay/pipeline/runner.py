"""Runs stages in order and guarantees that every stage that ran gets finalized exactly once.

The control flow mirrors nested ``async with`` blocks:

    async with log: async with route: async with provider: ...

- entry happens in order;
- exit happens in reverse, whatever happens: a return, a rejection, an exception, or
  cancellation (the client hung up);
- the outermost stage exits last, so it sees the final outcome. That is why the log stage
  goes first.

**Streams change who exits.** When the provider stage short-circuits with a stream, the
request is not over: the tokens haven't been sent yet. Finalizing here would log a request
with no usage and the wrong outcome, and (Phase 6) reconcile a reservation before a single
token was counted. So ``run()`` does *not* finalize. It returns a ``StreamResult`` holding a
``Finalizer``, and whoever holds it (the streaming response) must call it once the stream
ends. This is an explicit transfer of ownership: exactly one party is responsible for cleanup
at any moment.

A plain loop with ``finally`` is used instead of ``contextlib.AsyncExitStack`` so the order,
the outcome and the handoff are all visible in one place.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from prometheus_client import Counter

from relay.pipeline.context import RequestContext
from relay.pipeline.stage import Continue, Outcome, Reject, ShortCircuit, Stage
from relay.pipeline.streaming import PrimedStream
from relay.schemas.chat import ChatResponse
from relay.schemas.errors import ErrorClass, RelayError

logger = logging.getLogger(__name__)

FINALIZE_ERRORS = Counter(
    "relay_stage_finalize_errors_total",
    "finalize() calls that raised; the error was logged and swallowed.",
    ["stage"],  # stage names are a fixed, small set, so this label is bounded
)


class Finalizer:
    """Finalizes a set of stages exactly once. A second call is a no-op.

    Once-only matters because cleanup paths overlap. For example, a stream can fail *and*
    then be cancelled. If finalize ran twice, the log would get two rows and a reservation
    would be released twice.
    """

    def __init__(self, ran: Sequence[Stage], ctx: RequestContext) -> None:
        self._ran = tuple(ran)
        self._ctx = ctx
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    async def __call__(self, outcome: Outcome) -> None:
        if self._done:
            return
        self._done = True
        for stage in reversed(self._ran):
            try:
                await stage.finalize(self._ctx, outcome)
            except Exception:
                # Cleanup must never turn a served response into an error, or stop the other
                # stages from cleaning up. Count it so it shows up on a dashboard.
                # stdlib logging is synchronous I/O, which is acceptable only because this
                # path is rare. Per-request rows go through the async LogWriter.
                FINALIZE_ERRORS.labels(stage=stage.name).inc()
                logger.exception(
                    "finalize failed: stage=%s request_id=%s", stage.name, self._ctx.request_id
                )


@dataclass(frozen=True)
class StreamResult:
    stream: PrimedStream
    finalize: Finalizer  # the holder must call this exactly once, when the stream ends


class Pipeline:
    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages = tuple(stages)

    async def run(self, ctx: RequestContext) -> ChatResponse | StreamResult:
        ran: list[Stage] = []
        # Every path below overwrites this. The default only matters if something goes
        # wrong so early that no branch assigns it, and then "internal error" is honest.
        outcome = Outcome("error", ErrorClass.INTERNAL_ERROR)
        handed_off = False
        try:
            for stage in self._stages:
                # Recorded *before* process(): a stage that raises halfway may already hold
                # something (a reservation, a slot), so it must still be finalized. Unlike
                # __aexit__, finalize() must therefore cope with partial state.
                ran.append(stage)
                match await stage.process(ctx):
                    case Continue():
                        continue
                    case ShortCircuit(response=PrimedStream() as stream):
                        handed_off = True
                        return StreamResult(stream, Finalizer(ran, ctx))
                    case ShortCircuit(response=ChatResponse() as response):
                        outcome = Outcome("success")
                        return response
                    case Reject(error=error):
                        raise error
            raise RuntimeError("pipeline finished but no stage produced a response")
        except asyncio.CancelledError:
            outcome = Outcome("cancelled")
            raise  # swallowing CancelledError breaks asyncio's cancellation; always re-raise
        except RelayError as e:
            outcome = Outcome("error", e.error_class)
            raise
        except Exception:
            outcome = Outcome("error", ErrorClass.INTERNAL_ERROR)
            raise
        finally:
            if not handed_off:
                await Finalizer(ran, ctx)(outcome)
