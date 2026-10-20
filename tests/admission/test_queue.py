import asyncio

import pytest

from relay.admission.queue import AdmissionQueue, QueueFull, QueueTimeout


def _queue(interactive: float = 5.0, batch: float = 5.0, depth: int = 100) -> AdmissionQueue:
    return AdmissionQueue(max_wait_s={"interactive": interactive, "batch": batch}, max_depth=depth)


async def test_fast_path_when_a_slot_is_free() -> None:
    q = _queue()
    slot = await q.acquire("t", limit=2, priority="interactive")
    assert slot.waited_s == 0
    assert q.active("t") == 1
    q.release(slot, limit=2)
    assert q.active("t") == 0


async def test_waiters_get_slots_in_stride_order() -> None:
    q = _queue()
    held = await q.acquire("t", limit=1, priority="interactive")
    order: list[str] = []

    async def wait(name: str, priority: str) -> None:
        slot = await q.acquire("t", limit=1, priority=priority)  # type: ignore[arg-type]
        order.append(name)
        await asyncio.sleep(0)
        q.release(slot, limit=1)

    tasks = [asyncio.create_task(wait(f"b{i}", "batch")) for i in range(2)]
    tasks += [asyncio.create_task(wait(f"i{i}", "interactive")) for i in range(8)]
    await asyncio.sleep(0)  # let them all queue up
    q.release(held, limit=1)
    await asyncio.gather(*tasks)
    # With both classes backlogged: about four interactive per batch, and batch is never
    # starved even though it is outnumbered 4:1. FIFO within each class.
    assert order[:5].count("b0") + order[:6].count("b0") >= 1
    assert [n for n in order if n.startswith("i")] == [f"i{i}" for i in range(8)]
    assert [n for n in order if n.startswith("b")] == ["b0", "b1"]
    assert q.active("t") == 0


async def test_timeout_raises_and_leaves_no_trace() -> None:
    q = _queue(interactive=0.05)
    held = await q.acquire("t", limit=1, priority="interactive")
    with pytest.raises(QueueTimeout):
        await q.acquire("t", limit=1, priority="interactive")
    assert q.waiting("t") == 0
    q.release(held, limit=1)
    assert q.active("t") == 0  # the timed-out waiter didn't get (or leak) a slot


async def test_full_queue_refuses_immediately() -> None:
    q = _queue(depth=1)
    held = await q.acquire("t", limit=1, priority="interactive")
    waiter = asyncio.create_task(q.acquire("t", limit=1, priority="interactive"))
    await asyncio.sleep(0)
    with pytest.raises(QueueFull):
        await q.acquire("t", limit=1, priority="interactive")
    q.release(held, limit=1)
    q.release(await waiter, limit=1)


async def test_cancelled_waiter_is_removed() -> None:
    q = _queue()
    held = await q.acquire("t", limit=1, priority="interactive")
    waiter = asyncio.create_task(q.acquire("t", limit=1, priority="interactive"))
    await asyncio.sleep(0)
    waiter.cancel()  # the client disconnected while queued
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert q.waiting("t") == 0
    q.release(held, limit=1)
    assert q.active("t") == 0


async def test_slot_granted_at_the_moment_of_cancellation_is_passed_on() -> None:
    # The race: release() hands the slot to a waiter, and that waiter is cancelled before it
    # resumes. The slot must go to the next waiter, not leak.
    q = _queue()
    held = await q.acquire("t", limit=1, priority="interactive")
    first = asyncio.create_task(q.acquire("t", limit=1, priority="interactive"))
    second = asyncio.create_task(q.acquire("t", limit=1, priority="interactive"))
    await asyncio.sleep(0)
    q.release(held, limit=1)  # grants `first`'s future...
    first.cancel()  # ...which is cancelled before it runs again
    with pytest.raises(asyncio.CancelledError):
        await first
    slot = await second  # the slot moved on
    q.release(slot, limit=1)
    assert q.active("t") == 0


async def test_tenants_are_independent() -> None:
    q = _queue(interactive=0.05)
    await q.acquire("a", limit=1, priority="interactive")
    slot = await q.acquire("b", limit=1, priority="interactive")  # not blocked by tenant a
    assert slot.waited_s == 0
