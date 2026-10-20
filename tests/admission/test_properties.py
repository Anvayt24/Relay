"""Property-based tests for admission (CLAUDE.md: budget/admission code requires them).

Example tests check the cases we thought of. Here hypothesis *generates* thousands of
operation sequences (reserves, settles, double settles, clock jumps, expiries) and checks that
the rules hold after every single step. When it finds a failure, it shrinks it to the
shortest sequence that still fails.

The properties, from PRD FR-4's acceptance criteria:
1. **Budget:** at every admission decision, ``committed + outstanding + amount ≤ limit``, and
   the store agrees with a simple model of what it should contain.
2. **Reconcile:** at the end, nothing is left outstanding; every reservation was settled,
   released or expired.
3. **No starvation:** under contention, batch is served at least once per 5 interactive
   grants, and interactive is never stuck behind more than 1 batch grant.
4. **Token bucket:** it never grants more than ``capacity + rate * elapsed``.
"""

from collections.abc import Coroutine
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from relay.admission.queue import StridePicker
from relay.admission.store import MemoryStore


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Drive a MemoryStore coroutine to completion without an event loop.

    MemoryStore methods never actually await anything, so the first ``send`` runs them to the
    end. That keeps thousands of hypothesis steps fast, with no loop setup per step. If a
    method ever did await, this raises instead of silently misbehaving."""
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value  # type: ignore[no-any-return]
    raise RuntimeError("MemoryStore awaited something; it must stay synchronous")


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


LIMIT = 1_000


class BudgetMachine(RuleBasedStateMachine):
    """A random walk over one budget, checked against a dict-and-int model of it."""

    reservations = Bundle("reservations")

    @initialize()
    def setup(self) -> None:
        self.clock = Clock()
        self.store = MemoryStore(self.clock)
        self.committed = 0
        self.outstanding: dict[str, tuple[int, float]] = {}  # id -> (amount, expires)
        self.settled: set[str] = set()
        self.next_id = 0

    def _expire(self) -> None:
        self.outstanding = {k: v for k, v in self.outstanding.items() if v[1] > self.clock.t}

    @rule(target=reservations, amount=st.integers(0, 600), ttl=st.floats(0.1, 10))
    def reserve(self, amount: int, ttl: float) -> str:
        self._expire()
        res_id = f"r{self.next_id}"
        self.next_id += 1
        held = sum(a for a, _ in self.outstanding.values())
        ok = run(self.store.reserve("b", res_id, amount=amount, limit=LIMIT, ttl_s=ttl))
        # THE invariant (FR-4): admitted only if it fits, and always admitted if it fits.
        assert ok == (self.committed + held + amount <= LIMIT)
        if ok:
            self.outstanding[res_id] = (amount, self.clock.t + ttl)
        return res_id

    @rule(res_id=reservations, actual=st.integers(0, 800))
    def settle(self, res_id: str, actual: int) -> None:
        # `reservations` may hand back an id that was already settled, refused or expired:
        # exactly the duplicates and stragglers that real finalizes can produce.
        first = res_id not in self.settled
        assert run(self.store.settle("b", res_id, actual=actual)) == first
        if first:
            self.settled.add(res_id)
            self.outstanding.pop(res_id, None)
            self.committed += actual

    @rule(dt=st.floats(0, 5))
    def advance(self, dt: float) -> None:
        self.clock.t += dt

    @invariant()
    def store_matches_model(self) -> None:
        if not hasattr(self, "store"):
            return
        self._expire()
        state = run(self.store.budget_state("b"))
        assert state.committed == self.committed
        assert state.outstanding == sum(a for a, _ in self.outstanding.values())

    def teardown(self) -> None:
        if not hasattr(self, "store"):
            return
        # Reconcile everything still open: nothing may remain outstanding afterwards.
        for res_id in list(self.outstanding):
            run(self.store.settle("b", res_id, actual=0))
        assert run(self.store.budget_state("b")).outstanding == 0


TestBudget = BudgetMachine.TestCase
TestBudget.settings = settings(max_examples=200, stateful_step_count=40, deadline=None)


@given(st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=300))
def test_stride_picker_never_starves_either_class(arrivals: list[tuple[bool, bool]]) -> None:
    """Each step, maybe an interactive and/or batch request arrives, then one is granted."""
    picker = StridePicker()
    waiting = {"interactive": 0, "batch": 0}
    run_class, run_length = None, 0
    for new_interactive, new_batch in arrivals:
        for cls, arrived in (("interactive", new_interactive), ("batch", new_batch)):
            if arrived:
                if waiting[cls] == 0:
                    picker.activate(cls)  # type: ignore[arg-type]
                waiting[cls] += 1
        nonempty = {c for c, n in waiting.items() if n > 0}
        if not nonempty:
            run_class, run_length = None, 0
            continue
        both_waiting = len(nonempty) == 2
        chosen = picker.pick(nonempty)  # type: ignore[arg-type]
        assert chosen in nonempty  # never grants a class with nobody waiting
        waiting[chosen] -= 1
        run_length = run_length + 1 if chosen == run_class else 1
        run_class = chosen
        if both_waiting:
            # While the other class waits, this class's run is bounded: 4:1 weights.
            assert run_length <= (5 if chosen == "interactive" else 1)


def test_stride_picker_shares_4_to_1_under_backlog() -> None:
    picker = StridePicker()
    grants = [picker.pick({"interactive", "batch"}) for _ in range(1000)]
    assert grants.count("batch") == 200  # exactly 1/5: guaranteed, not just on average


@given(
    st.lists(st.tuples(st.floats(0, 2), st.integers(1, 30)), min_size=1, max_size=100),
    st.integers(1, 50),
    st.floats(0.5, 20),
)
@settings(deadline=None)
def test_token_bucket_never_grants_more_than_capacity_plus_refill(
    steps: list[tuple[float, int]], capacity: int, rate: float
) -> None:
    clock = Clock()
    store = MemoryStore(clock)
    granted = 0
    for dt, amount in steps:
        clock.t += dt
        if run(store.take("k", capacity=capacity, refill_per_s=rate, amount=amount)) == 0:
            granted += amount
        assert granted <= capacity + rate * clock.t + 1e-6
