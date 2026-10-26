"""FR-4: the USD hold covers the most expensive model the request might fall back to."""

from relay.admission.stages import ReservationStage
from relay.admission.store import MemoryStore

from .conftest import FALLBACK_CONFIG, make_ctx  # type: ignore[import-not-found]


async def _held(planner: object) -> tuple[int, int]:
    store = MemoryStore()
    stage = ReservationStage(store, FALLBACK_CONFIG, planner=planner)  # type: ignore[arg-type]
    ctx = make_ctx(route="deep")
    await stage.process(ctx)
    assert ctx.reservation is not None
    return ctx.reservation.nano, ctx.reservation.tokens


async def test_reservation_is_sized_for_the_priciest_candidate() -> None:
    primary_only, primary_tokens = await _held(lambda ctx: ["p/model"])
    worst, worst_tokens = await _held(lambda ctx: ["p/model", "b/model", "big/model"])
    # big/model: twice the price and a 4096-token default cap versus 1024.
    assert worst > primary_only * 4
    assert worst_tokens > primary_tokens


async def test_settle_charges_only_what_the_serving_model_cost() -> None:
    store = MemoryStore()
    stage = ReservationStage(store, FALLBACK_CONFIG, planner=lambda c: ["p/model", "big/model"])
    ctx = make_ctx(route="deep")
    await stage.process(ctx)
    assert ctx.reservation is not None
    budget = ctx.reservation.budget
    before = await store.budget_state(budget)
    assert before.outstanding == ctx.reservation.nano
    from relay.pipeline.stage import Outcome

    await stage.finalize(ctx, Outcome("error"))
    after = await store.budget_state(budget)
    assert after.outstanding == 0 and after.committed == 0  # nothing ran: fully released
