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


async def test_a_streams_hold_outlives_its_deadline() -> None:
    # Found in review: the hold expired at deadline + 60 s, but a stream isn't bound by the
    # deadline after its first byte, so a long stream's money could be reserved by others.
    from relay.admission.stages import STREAM_HOLD_S

    ttls: dict[bool, float] = {}

    class Spy(MemoryStore):
        async def reserve(self, budget, res_id, *, amount, limit, ttl_s):  # type: ignore[no-untyped-def]
            ttls[stream] = ttl_s
            return await super().reserve(budget, res_id, amount=amount, limit=limit, ttl_s=ttl_s)

    for stream in (False, True):
        stage = ReservationStage(Spy(), FALLBACK_CONFIG, planner=lambda c: ["p/model"])
        await stage.process(make_ctx(route="deep", stream=stream, deadline_s=60))
    assert ttls[False] <= 121
    assert ttls[True] >= STREAM_HOLD_S
