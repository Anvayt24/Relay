"""The circuit breaker state machine (PRD FR-5; ADR 0009)."""

from hypothesis import given, settings
from hypothesis import strategies as st

from relay.config.schema import ReliabilityConfig
from relay.reliability.breaker import BreakerState, CircuitBreaker, Permit
from relay.schemas.errors import ErrorClass

from .conftest import FakeClock  # type: ignore[import-not-found]

CFG = ReliabilityConfig()  # PRD defaults: 50% of >= 10 calls in 30 s, open 30 s, 3 probes
FAIL = ErrorClass.SERVER_ERROR


def breaker(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker("p", "m", CFG, clock)


def call(b: CircuitBreaker, outcome: ErrorClass | None) -> None:
    permit = b.acquire()
    assert permit is not None
    b.record(permit, outcome)


def trip(b: CircuitBreaker) -> None:
    for _ in range(10):
        call(b, FAIL)
    assert b.state is BreakerState.OPEN


def test_opens_at_half_failures_once_there_are_enough_calls() -> None:
    b = breaker(FakeClock())
    for _ in range(9):
        call(b, FAIL)
    assert b.state is BreakerState.CLOSED  # 9 calls: below the minimum, however bad
    call(b, None)
    assert b.state is BreakerState.OPEN  # 9 of 10 failed
    assert b.acquire() is None


def test_stays_closed_below_the_failure_ratio() -> None:
    b = breaker(FakeClock())
    for i in range(40):
        call(b, FAIL if i % 3 == 0 else None)  # one in three fails
    assert b.state is BreakerState.CLOSED


def test_old_failures_fall_out_of_the_rolling_window() -> None:
    clock = FakeClock()
    b = breaker(clock)
    for _ in range(9):
        call(b, FAIL)
    clock.advance(31)  # those nine are now outside the 30 s window
    call(b, FAIL)
    assert b.state is BreakerState.CLOSED


def test_rate_limits_neither_trip_nor_dilute() -> None:
    b = breaker(FakeClock())
    for _ in range(50):
        call(b, ErrorClass.RATE_LIMITED)
    assert b.state is BreakerState.CLOSED  # 429s are the pause state's job
    for _ in range(10):
        call(b, FAIL)
    # Had the 50 429s counted as successes, 10 failures out of 60 would not open it.
    assert b.state is BreakerState.OPEN


def test_half_open_admits_exactly_the_probe_count_then_closes() -> None:
    clock = FakeClock()
    b = breaker(clock)
    trip(b)
    assert b.retry_after_s() == 30
    clock.advance(30)
    assert b.state is BreakerState.HALF_OPEN
    probes = [b.acquire() for _ in range(5)]
    held = [p for p in probes if p is not None]
    assert len(held) == 3 and all(p.probe for p in held)
    for p in held:
        b.record(p, None)
    assert b.state is BreakerState.CLOSED


def test_a_failed_probe_reopens() -> None:
    clock = FakeClock()
    b = breaker(clock)
    trip(b)
    clock.advance(30)
    probe = b.acquire()
    assert probe is not None
    b.record(probe, ErrorClass.TIMEOUT)
    assert b.state is BreakerState.OPEN
    assert b.retry_after_s() == 30


def test_a_probe_without_a_verdict_frees_its_slot() -> None:
    clock = FakeClock()
    b = breaker(clock)
    trip(b)
    clock.advance(30)
    held = [b.acquire() for _ in range(3)]
    assert b.acquire() is None
    first = held[0]
    assert first is not None
    b.record(first, ErrorClass.RATE_LIMITED)  # a 429 says nothing about health
    assert b.state is BreakerState.HALF_OPEN
    assert b.acquire() is not None  # the slot came back
    b.release(Permit("p", "m", probe=True))  # an unused permit also gives its slot back
    assert b.acquire() is not None


@settings(max_examples=300, deadline=None)
@given(
    ops=st.lists(
        st.one_of(
            st.tuples(st.just("acquire"), st.just(0.0)),
            st.tuples(
                st.just("record"),
                st.sampled_from([0.0, 1.0, 2.0, 3.0]),  # index into OUTCOMES
            ),
            st.tuples(st.just("tick"), st.floats(0, 40)),
        ),
        max_size=120,
    )
)
def test_breaker_invariants_hold_for_any_history(ops: list[tuple[str, float]]) -> None:
    """For any interleaving of calls, outcomes and time:
    - a half-open breaker never has more than `probes` permits out at once;
    - it never opens without `min_calls` counted calls in its window;
    - 429s alone never open it.
    """
    outcomes = [None, FAIL, ErrorClass.RATE_LIMITED, ErrorClass.TIMEOUT]
    clock = FakeClock()
    b = breaker(clock)
    held: list[Permit] = []
    counted_since_close: list[float] = []  # times of counted calls while closed
    for op, arg in ops:
        if op == "acquire":
            permit = b.acquire()
            if permit is not None:
                held.append(permit)
        elif op == "record" and held:
            permit = held.pop(0)
            outcome = outcomes[int(arg)]
            was = b.state
            b.record(permit, outcome)
            if (
                was is BreakerState.CLOSED
                and not permit.probe
                and outcome is not ErrorClass.RATE_LIMITED
            ):
                counted_since_close.append(clock.now)
            if was is BreakerState.CLOSED and b.state is BreakerState.OPEN:
                recent = [t for t in counted_since_close if t > clock.now - 31]
                assert len(recent) >= CFG.breaker_min_calls
                assert outcome is not ErrorClass.RATE_LIMITED
            if b.state is BreakerState.CLOSED and was is not BreakerState.CLOSED:
                counted_since_close.clear()
        elif op == "tick":
            clock.advance(arg)
        if b.state is BreakerState.HALF_OPEN:
            assert sum(1 for p in held if p.probe) <= CFG.breaker_half_open_probes
