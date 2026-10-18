import asyncio

import pytest
from prometheus_client import REGISTRY

from relay.pipeline.context import RequestContext
from relay.pipeline.runner import Pipeline
from relay.pipeline.stage import Continue, Outcome, Reject, ShortCircuit, StageResult
from relay.schemas.chat import ChatResponse
from relay.schemas.errors import ErrorClass, RelayError


class Recorder:
    """Shared log of what every fake stage did, in order."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.outcomes: dict[str, Outcome] = {}


class FakeStage:
    """A stage whose process() returns a scripted result (or raises a scripted exception).

    It satisfies the Stage protocol structurally: no base class, no registration. That is
    the point of using a Protocol.
    """

    def __init__(
        self,
        name: str,
        recorder: Recorder,
        result: StageResult | None = None,
        raises: BaseException | None = None,
        finalize_raises: Exception | None = None,
    ) -> None:
        self.name = name
        self._rec = recorder
        self._result = result or Continue()
        self._raises = raises
        self._finalize_raises = finalize_raises

    async def process(self, ctx: RequestContext) -> StageResult:
        self._rec.events.append(f"process:{self.name}")
        if self._raises is not None:
            raise self._raises
        return self._result

    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        self._rec.events.append(f"finalize:{self.name}")
        self._rec.outcomes[self.name] = outcome
        if self._finalize_raises is not None:
            raise self._finalize_raises


async def test_stages_run_in_order_and_finalize_in_reverse(
    ctx: RequestContext, chat_response: ChatResponse
) -> None:
    rec = Recorder()
    pipeline = Pipeline(
        [
            FakeStage("a", rec),
            FakeStage("b", rec),
            FakeStage("c", rec, result=ShortCircuit(chat_response)),
        ]
    )

    response = await pipeline.run(ctx)

    assert response is chat_response
    assert rec.events == [
        "process:a",
        "process:b",
        "process:c",
        "finalize:c",
        "finalize:b",
        "finalize:a",
    ]
    assert all(o == Outcome("success") for o in rec.outcomes.values())


async def test_short_circuit_skips_later_stages_entirely(
    ctx: RequestContext, chat_response: ChatResponse
) -> None:
    rec = Recorder()
    pipeline = Pipeline(
        [
            FakeStage("cache", rec, result=ShortCircuit(chat_response)),
            FakeStage("provider", rec),
        ]
    )

    await pipeline.run(ctx)

    # The provider stage never ran, so it must not be finalized either: finalizing a stage
    # that never acquired anything would e.g. release a reservation that was never made.
    assert rec.events == ["process:cache", "finalize:cache"]


async def test_reject_raises_and_finalizes_with_error_outcome(ctx: RequestContext) -> None:
    rec = Recorder()
    error = RelayError(404, ErrorClass.MODEL_NOT_FOUND, "no such model")
    pipeline = Pipeline(
        [
            FakeStage("log", rec),
            FakeStage("route", rec, result=Reject(error)),
            FakeStage("provider", rec),
        ]
    )

    with pytest.raises(RelayError) as exc:
        await pipeline.run(ctx)

    assert exc.value is error
    assert rec.events == ["process:log", "process:route", "finalize:route", "finalize:log"]
    assert rec.outcomes["log"] == Outcome("error", ErrorClass.MODEL_NOT_FOUND)


async def test_unexpected_exception_finalizes_including_the_failing_stage(
    ctx: RequestContext,
) -> None:
    rec = Recorder()
    pipeline = Pipeline(
        [
            FakeStage("log", rec),
            FakeStage("buggy", rec, raises=ValueError("bug")),
        ]
    )

    with pytest.raises(ValueError, match="bug"):
        await pipeline.run(ctx)

    # The stage that raised did start, so it may hold partial state: it is finalized too.
    assert rec.events == ["process:log", "process:buggy", "finalize:buggy", "finalize:log"]
    assert rec.outcomes["log"] == Outcome("error", ErrorClass.INTERNAL_ERROR)


async def test_cancellation_finalizes_with_cancelled_and_reraises(ctx: RequestContext) -> None:
    rec = Recorder()
    entered = asyncio.Event()

    class Hangs(FakeStage):
        async def process(self, ctx: RequestContext) -> StageResult:
            self._rec.events.append(f"process:{self.name}")
            entered.set()
            await asyncio.sleep(3600)
            return Continue()  # pragma: no cover

    pipeline = Pipeline([FakeStage("log", rec), Hangs("provider", rec)])
    task = asyncio.create_task(pipeline.run(ctx))
    await entered.wait()
    task.cancel()  # what happens when a client disconnects

    with pytest.raises(asyncio.CancelledError):
        await task

    assert rec.events[-2:] == ["finalize:provider", "finalize:log"]
    assert rec.outcomes["log"] == Outcome("cancelled")


async def test_failing_finalize_is_contained(
    ctx: RequestContext, chat_response: ChatResponse
) -> None:
    rec = Recorder()
    pipeline = Pipeline(
        [
            FakeStage("log", rec),
            FakeStage("flaky", rec, finalize_raises=RuntimeError("cleanup failed")),
            FakeStage("provider", rec, result=ShortCircuit(chat_response)),
        ]
    )
    before = _finalize_errors("flaky")

    response = await pipeline.run(ctx)

    # The response still goes out, and the stages after the broken one still finalize.
    assert response is chat_response
    assert rec.events[-3:] == ["finalize:provider", "finalize:flaky", "finalize:log"]
    assert _finalize_errors("flaky") == before + 1


async def test_pipeline_without_a_responding_stage_is_a_bug(ctx: RequestContext) -> None:
    rec = Recorder()
    pipeline = Pipeline([FakeStage("a", rec), FakeStage("b", rec)])

    with pytest.raises(RuntimeError, match="no stage"):
        await pipeline.run(ctx)

    assert rec.events[-2:] == ["finalize:b", "finalize:a"]


def _finalize_errors(stage: str) -> float:
    value = REGISTRY.get_sample_value("relay_stage_finalize_errors_total", {"stage": stage})
    return value or 0.0
