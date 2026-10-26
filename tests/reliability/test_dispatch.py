"""The retry owner end to end, at the dispatcher (PRD FR-5, FR-4; ADR 0008).

Each test scripts the providers behind a fallback graph and asserts on two things: what the
client gets, and the exact upstream calls made (``provider.calls``). The second is the real
contract of a retry owner: it is as much about the calls it does *not* make.
"""

import pytest

from relay.admission.store import MemoryStore
from relay.pipeline.streaming import PrimedStream
from relay.providers.base import ProviderAdapter
from relay.reliability.dispatch import ReliableDispatcher
from relay.schemas.chat import ChatResponse
from relay.schemas.errors import ErrorClass, RelayError
from relay.simulator.plan import MockBehavior

from ..conftest import build_dispatcher  # type: ignore[import-not-found]
from .conftest import (  # type: ignore[import-not-found]
    FALLBACK_CONFIG,
    FakeClock,
    ScriptedProvider,
    make_ctx,
)


def providers(**overrides: ScriptedProvider) -> dict[str, ScriptedProvider]:
    base = {name: ScriptedProvider() for name in ("primary", "backup", "big", "capped")}
    return {**base, **overrides}


def dispatcher(
    provs: dict[str, ScriptedProvider], *, clock: FakeClock | None = None, **kw: object
) -> ReliableDispatcher:
    adapters: dict[str, ProviderAdapter] = dict(provs)
    if clock is None:
        return build_dispatcher(FALLBACK_CONFIG, adapters, **kw)  # type: ignore[arg-type]
    return build_dispatcher(FALLBACK_CONFIG, adapters, clock=clock, **kw)  # type: ignore[arg-type]


def fail(error: ErrorClass, **kw: object) -> ScriptedProvider:
    return ScriptedProvider(then=MockBehavior(error=error, **kw))  # type: ignore[arg-type]


async def test_transient_error_is_retried_once_then_falls_back() -> None:
    provs = providers(primary=fail(ErrorClass.SERVER_ERROR))
    ctx = make_ctx(route="safe")
    result = await dispatcher(provs).run(ctx)

    assert isinstance(result, ChatResponse)
    assert result.model == "b/model"  # Relay's name for the model that answered
    assert provs["primary"].calls == 2  # the call and its one retry
    assert provs["backup"].calls == 1
    assert [(a.model, a.hop, a.retry, a.error_class) for a in ctx.attempts] == [
        ("p/model", 0, 0, ErrorClass.SERVER_ERROR),
        ("p/model", 0, 1, ErrorClass.SERVER_ERROR),
        ("b/model", 1, 0, None),
    ]
    assert ctx.served is not None and ctx.served.provider == "backup"
    assert ctx.pricing is not None and ctx.pricing.model is FALLBACK_CONFIG.models["b/model"]
    assert not ctx.degraded  # a `safe` edge


async def test_a_retry_that_succeeds_needs_no_fallback() -> None:
    provs = providers(primary=ScriptedProvider(MockBehavior(error=ErrorClass.OVERLOADED)))
    ctx = make_ctx(route="safe")
    result = await dispatcher(provs).run(ctx)
    assert isinstance(result, ChatResponse) and result.model == "p/model"
    assert provs["primary"].calls == 2 and provs["backup"].calls == 0


async def test_rate_limit_pauses_and_falls_back_without_retrying() -> None:
    provs = providers(primary=fail(ErrorClass.RATE_LIMITED, retry_after_s=30))
    d = dispatcher(provs)
    ctx = make_ctx(route="safe")
    assert isinstance(await d.run(ctx), ChatResponse)
    assert provs["primary"].calls == 1  # 429s are never retried on the same limit

    # The next request doesn't even try the paused primary: zero calls, straight to backup.
    ctx2 = make_ctx(route="safe")
    assert isinstance(await d.run(ctx2), ChatResponse)
    assert provs["primary"].calls == 1
    assert provs["backup"].calls == 2
    assert ctx2.attempts[0].skipped_reason == "paused"


async def test_paused_with_no_fallback_keeps_the_client_retryable_429() -> None:
    provs = providers(primary=fail(ErrorClass.RATE_LIMITED, retry_after_s=30))
    d = dispatcher(provs)
    with pytest.raises(RelayError):
        await d.run(make_ctx(route="none"))
    with pytest.raises(RelayError) as exc:
        await d.run(make_ctx(route="none"))
    assert exc.value.status == 429
    assert exc.value.error_class is ErrorClass.PROVIDER_PAUSED
    assert exc.value.retry_after_s is not None and exc.value.retry_after_s >= 30
    assert provs["primary"].calls == 1


async def test_timeout_before_first_token_falls_back_for_streams() -> None:
    provs = providers(primary=ScriptedProvider(then=MockBehavior(first_chunk_delay_s=5)))
    ctx = make_ctx(route="safe", stream=True)
    result = await dispatcher(provs, first_token_ceiling_s=0.05).run(ctx)
    assert isinstance(result, PrimedStream)
    assert result.relay_model == "b/model"
    assert ctx.attempts[0].error_class is ErrorClass.TIMEOUT
    assert provs["primary"].calls == 1  # a timeout is not retried on the same model
    await result.upstream.aclose()


async def test_primed_stream_is_the_commit_point_and_the_permit_waits() -> None:
    # A stream that primes fine is "served" by that model; what happens after the first
    # byte (mid-stream drop) is the response's business, never a fallback. Its breaker
    # verdict is deferred to finalize via ctx.breaker_permit.
    provs = providers(primary=ScriptedProvider(then=MockBehavior(fail_after_chunks=1)))
    ctx = make_ctx(route="safe", stream=True)
    result = await dispatcher(provs).run(ctx)
    assert isinstance(result, PrimedStream) and result.relay_model == "p/model"
    assert ctx.breaker_permit is not None
    assert provs["backup"].calls == 0
    await result.upstream.aclose()


async def test_context_length_falls_back_only_to_a_larger_window() -> None:
    provs = providers(primary=fail(ErrorClass.CONTEXT_LENGTH_EXCEEDED))
    ctx = make_ctx(route="context")
    result = await dispatcher(provs).run(ctx)
    assert isinstance(result, ChatResponse) and result.model == "big/model"
    # b/model has the same 8k window as the primary: skipped, not called.
    assert provs["backup"].calls == 0
    assert [a.skipped_reason for a in ctx.attempts] == [None, "context_window", None]


async def test_terminal_errors_are_returned_without_fallback() -> None:
    provs = providers(primary=fail(ErrorClass.CONTENT_FILTERED))
    with pytest.raises(RelayError) as exc:
        await dispatcher(provs).run(make_ctx(route="safe"))
    assert exc.value.error_class is ErrorClass.CONTENT_FILTERED
    assert provs["backup"].calls == 0  # trying other models could launder a moderation block


async def test_strict_with_only_undecided_edges_is_503() -> None:
    provs = providers(primary=fail(ErrorClass.SERVER_ERROR))
    ctx = make_ctx(route="gated")  # tenant `strict`
    with pytest.raises(RelayError) as exc:
        await dispatcher(provs).run(ctx)
    # The undecided edge isn't a candidate at all, so no fallback existed: the provider's
    # own class. (With a fallback that also failed it would be no_usable_fallback.)
    assert exc.value.error_class is ErrorClass.SERVER_ERROR
    assert provs["backup"].calls == 0


async def test_all_candidates_failing_is_503_no_usable_fallback() -> None:
    provs = providers(primary=fail(ErrorClass.SERVER_ERROR), backup=fail(ErrorClass.OVERLOADED))
    with pytest.raises(RelayError) as exc:
        await dispatcher(provs).run(make_ctx(route="safe"))
    assert exc.value.status == 503
    assert exc.value.error_class is ErrorClass.NO_USABLE_FALLBACK
    assert exc.value.retry_after_s is not None


async def test_allow_degraded_serves_through_an_undecided_edge_and_marks_it() -> None:
    provs = providers(primary=fail(ErrorClass.SERVER_ERROR))
    ctx = make_ctx(route="gated", tenant="degraded")
    result = await dispatcher(provs).run(ctx)
    assert isinstance(result, ChatResponse) and result.model == "b/model"
    assert ctx.degraded


async def test_explicit_model_gets_retries_but_no_fallback() -> None:
    provs = providers(primary=fail(ErrorClass.SERVER_ERROR))
    with pytest.raises(RelayError) as exc:
        await dispatcher(provs).run(make_ctx(route="safe", reason="explicit"))
    assert exc.value.error_class is ErrorClass.SERVER_ERROR
    assert provs["primary"].calls == 2 and provs["backup"].calls == 0


async def test_hop_limit_counts_attempted_fallbacks_only() -> None:
    provs = providers(
        primary=fail(ErrorClass.TIMEOUT),
        backup=fail(ErrorClass.TIMEOUT),
        capped=fail(ErrorClass.TIMEOUT),
    )
    with pytest.raises(RelayError):
        await dispatcher(provs).run(make_ctx(route="deep"))
    # deep: p -> b, c (depth 1), then big (depth 2). max_fallback_hops = 2, so b and c are
    # attempted and big never is.
    assert (provs["backup"].calls, provs["capped"].calls, provs["big"].calls) == (1, 1, 0)


async def test_deadline_stops_further_attempts_with_504() -> None:
    clock = FakeClock()
    provs = providers(primary=fail(ErrorClass.TIMEOUT))
    d = dispatcher(provs, clock=clock)
    ctx = make_ctx(route="safe", clock=clock, deadline_s=0.4)  # below min_attempt_s (0.5)
    with pytest.raises(RelayError) as exc:
        await d.run(ctx)
    assert exc.value.status == 504
    assert exc.value.error_class is ErrorClass.DEADLINE_EXCEEDED
    assert provs["primary"].calls == 0
    assert ctx.attempts[0].skipped_reason == "deadline"


async def test_open_breaker_is_skipped_without_a_call_then_recovers() -> None:
    clock = FakeClock()
    # The primary fails 10 times (opening its breaker), then is healthy again.
    primary = ScriptedProvider(*[MockBehavior(error=ErrorClass.TIMEOUT)] * 10)
    provs = providers(primary=primary)
    d = dispatcher(provs, clock=clock)
    for _ in range(10):
        await d.run(make_ctx(route="safe", clock=clock))
    assert primary.calls == 10

    ctx = make_ctx(route="safe", clock=clock)
    await d.run(ctx)
    assert primary.calls == 10  # breaker open: skipped, straight to backup
    assert ctx.attempts[0].skipped_reason == "breaker_open"

    clock.advance(30)  # half-open: probes go through, and they succeed
    for _ in range(3):
        result = await d.run(make_ctx(route="safe", clock=clock))
        assert isinstance(result, ChatResponse) and result.model == "p/model"
    assert primary.calls == 13


async def test_fallback_is_admitted_against_the_fallback_providers_capacity() -> None:
    # `capped` allows 1 request per minute. The first fallback uses it; the second request's
    # fallback skips it ("provider_capacity") and goes on to backup.
    provs = providers(primary=fail(ErrorClass.SERVER_ERROR))
    d = dispatcher(provs, store=MemoryStore())
    await d.run(make_ctx(route="capped"))
    ctx = make_ctx(route="capped")
    result = await d.run(ctx)
    assert isinstance(result, ChatResponse) and result.model == "b/model"
    assert provs["capped"].calls == 1
    assert "provider_capacity" in [a.skipped_reason for a in ctx.attempts]


async def test_model_not_found_marks_the_model_unavailable() -> None:
    provs = providers(primary=fail(ErrorClass.MODEL_NOT_FOUND))
    d = dispatcher(provs)
    await d.run(make_ctx(route="safe"))
    ctx = make_ctx(route="safe")
    await d.run(ctx)
    assert provs["primary"].calls == 1  # not called again while marked unavailable
    assert ctx.attempts[0].skipped_reason == "model_unavailable"


async def test_retry_budget_exhaustion_skips_straight_to_fallback() -> None:
    provs = providers(primary=fail(ErrorClass.SERVER_ERROR))
    d = dispatcher(provs)
    for _ in range(6):
        await d.run(make_ctx(route="safe"))
    # min 3 retries + 10% of 6 attempts: only the first 3 requests got a retry.
    assert provs["primary"].calls == 6 + 3
