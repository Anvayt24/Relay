"""Runs stages in order and guarantees that every stage that ran gets finalized.

The control flow mirrors nested ``async with`` blocks:

    async with log: async with route: async with provider: ...

- entry happens in order;
- exit happens in reverse, whatever happens: a return, a rejection, an exception, or
  cancellation (the client hung up);
- the outermost stage exits last, so it sees the final outcome. That is why the log stage
  will go first (Phase 2).

A plain loop with ``finally`` is used instead of ``contextlib.AsyncExitStack`` so the order
and the outcome passed to each stage are visible in one place.
"""

import asyncio
import logging
from collections.abc import Sequence

from prometheus_client import Counter

from relay.pipeline.context import RequestContext
from relay.pipeline.stage import Continue, Outcome, Reject, ShortCircuit, Stage
from relay.schemas.chat import ChatResponse
from relay.schemas.errors import ErrorClass, RelayError

logger = logging.getLogger(__name__)

FINALIZE_ERRORS = Counter(
    "relay_stage_finalize_errors_total",
    "finalize() calls that raised; the error was logged and swallowed.",
    ["stage"],  # stage names are a fixed, small set, so this label is bounded
)


class Pipeline:
    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages = tuple(stages)

    async def run(self, ctx: RequestContext) -> ChatResponse:
        ran: list[Stage] = []
        # Every path below overwrites this. The default only matters if something goes
        # wrong so early that no branch assigns it, and then "internal error" is honest.
        outcome = Outcome("error", ErrorClass.INTERNAL_ERROR)
        try:
            for stage in self._stages:
                # Recorded *before* process(): a stage that raises halfway may already hold
                # something (a reservation, a slot), so it must still be finalized. Unlike
                # __aexit__, finalize() must therefore cope with partial state.
                ran.append(stage)
                match await stage.process(ctx):
                    case Continue():
                        continue
                    case ShortCircuit(response=response):
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
            await self._finalize(ran, ctx, outcome)

    async def _finalize(self, ran: list[Stage], ctx: RequestContext, outcome: Outcome) -> None:
        for stage in reversed(ran):
            try:
                await stage.finalize(ctx, outcome)
            except Exception:
                # Cleanup must never turn a served response into an error, or stop the other
                # stages from cleaning up. Count it so it shows up on a dashboard.
                # stdlib logging is synchronous I/O, which is acceptable only because this
                # path is rare. Per-request logs go through the async writer (Phase 2).
                FINALIZE_ERRORS.labels(stage=stage.name).inc()
                logger.exception(
                    "finalize failed: stage=%s request_id=%s", stage.name, ctx.request_id
                )
