"""The single retry owner (PRD FR-5; ADR 0008): one loop that decides, for every request,
which model to call, whether to retry it, and when to fall back or give up.

    for each candidate (primary, then fallbacks; candidates.py):
        skip it if: marked unavailable · too small a context window · deadline too close ·
                    paused (429 elsewhere) · breaker open · provider out of capacity
        call it (a stream only up to its first content chunk; see pipeline/streaming.py)
        success → done
        failure → record it (breaker, pause, attempt row), then:
                  retry once if transient and the retry budget and deadline allow,
                  else fall back if Appendix B allows (policy.py), else stop

**The stream boundary.** For a stream, "success" means primed: the first content chunk has
arrived and nothing has been sent yet. Everything up to that point can fail over
transparently. After it, the response belongs to the client, and a failure is a mid-stream
failure, never a switch to another model (FR-5).

**Every skip and every call is recorded** in ``ctx.attempts`` (hop, retry, error class or
skip reason). That list is the ``attempts`` table, and it answers "why did this request end
up on that model?" for every request.
"""

import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from prometheus_client import Counter

from relay.admission.pause import PauseState
from relay.admission.store import CapacityStore, StoreUnavailable
from relay.admission.tokens import output_cap, prompt_tokens
from relay.config.models import ModelSpec
from relay.config.pricing import Pricing
from relay.config.schema import RelayConfig
from relay.pipeline.context import ProviderAttempt, RequestContext
from relay.pipeline.streaming import PrimedStream, prime
from relay.providers.base import ProviderAdapter
from relay.reliability.backoff import full_jitter
from relay.reliability.breaker import BreakerRegistry, BreakerState, Permit
from relay.reliability.candidates import Candidate, plan_candidates
from relay.reliability.deadline import Deadline
from relay.reliability.edges import EdgeStateSource
from relay.reliability.policy import CONTEXT_FALLBACK, RETRY_ONCE, can_fall_back
from relay.reliability.retry_budget import RetryBudget
from relay.schemas.chat import ChatRequest, ChatResponse
from relay.schemas.errors import ErrorClass, ProviderError, RelayError

ATTEMPTS = Counter(
    "relay_provider_attempts_total",
    "Upstream calls by outcome (error_class 'ok' = success).",
    ["provider", "model", "error_class"],
)
RETRIES = Counter("relay_retries_total", "Same-model retries sent upstream.", ["provider"])
FALLBACKS = Counter(
    "relay_fallbacks_total",
    "Requests served by a fallback model, by what made the previous candidate unusable.",
    ["route", "reason"],
)
DEGRADED = Counter(
    "relay_degraded_responses_total",
    "Responses served through an edge without evidence (X-Relay-Degraded).",
    ["route"],
)
MODEL_UNAVAILABLE = Counter(
    "relay_model_unavailable_total",
    "model_not_found from upstream: the model was skipped for a while. Alert on this.",
    ["provider", "model"],
)


@dataclass
class _Run:
    """What one request's dispatch has learned so far."""

    last_error: ProviderError | None = None
    stop: bool = False  # a failure that must not fall back (Appendix B)
    deadline_hit: bool = False
    # After context_length_exceeded: only models with a window larger than this.
    context_floor: float | None = None
    # How long until skipped candidates come back, for Retry-After.
    waits: list[float] = field(default_factory=list)
    # Why the previous candidate was left: the label on relay_fallbacks_total.
    left_because: str | None = None


class ReliableDispatcher:
    def __init__(
        self,
        config: RelayConfig,
        models: Mapping[str, ModelSpec],
        providers: Mapping[str, ProviderAdapter],
        *,
        store: CapacityStore,
        pause: PauseState | None,
        breakers: BreakerRegistry,
        retry_budget: RetryBudget,
        edge_states: EdgeStateSource,
        first_token_ceiling_s: float,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        missing = {spec.provider for spec in models.values()} - set(providers)
        if missing:
            raise ValueError(f"no adapter registered for provider(s): {sorted(missing)}")
        self._config = config
        self._cfg = config.reliability
        self._models = models
        self._providers = providers
        self._store = store
        self._pause = pause
        self._breakers = breakers
        self._budget = retry_budget
        self._edge_states = edge_states
        # The operator's ceiling on any first-token wait; the tenant's deadline can only
        # shorten it.
        self._first_token_ceiling_s = first_token_ceiling_s
        self._rng = rng or random.Random()
        self._clock = clock
        self._sleep = sleep
        # (provider, model) -> skip until (after model_not_found). Per replica.
        self._unavailable: dict[tuple[str, str], float] = {}

    @property
    def breakers(self) -> BreakerRegistry:
        return self._breakers

    def plan(self, ctx: RequestContext) -> list[Candidate]:
        if ctx.decision is None:
            raise RuntimeError("dispatch needs a route decision; RouteStage goes first")
        return plan_candidates(
            config=self._config,
            models=self._models,
            route=ctx.route,
            tenant=ctx.tenant,
            decision=ctx.decision,
            edge_states=self._edge_states,
        )

    async def run(self, ctx: RequestContext) -> ChatResponse | PrimedStream:
        """The response (or primed stream) of the first candidate that works. Raises
        ``RelayError`` when none does."""
        candidates = self.plan(ctx)
        max_hops = self._config.routes[ctx.route].max_fallback_hops
        deadline = Deadline(ctx.deadline)
        run = _Run()
        hops = 0
        for candidate in candidates:
            if candidate.is_fallback and hops >= max_hops:
                break
            spec = self._models[candidate.model]
            hop = hops + 1 if candidate.is_fallback else 0
            permit, reason = await self._admit(ctx, candidate, spec, run, deadline)
            if permit is None:
                ctx.attempts.append(
                    ProviderAttempt(spec.provider, candidate.model, 0.0, None, hop, 0, reason)
                )
                run.left_because = reason
                if run.deadline_hit:
                    break
                continue
            hops = hop
            result = await self._call(ctx, candidate, spec, hop, permit, run, deadline)
            if result is not None:
                if candidate.is_fallback:
                    FALLBACKS.labels(route=ctx.route, reason=run.left_because or "unknown").inc()
                if candidate.degraded:
                    ctx.degraded = True
                    DEGRADED.labels(route=ctx.route).inc()
                return result
            if run.stop or run.deadline_hit:
                break
        raise self._give_up(ctx, candidates, run)

    # ------------------------------------------------------------------ per candidate
    async def _admit(
        self,
        ctx: RequestContext,
        candidate: Candidate,
        spec: ModelSpec,
        run: _Run,
        deadline: Deadline,
    ) -> tuple[Permit | None, str | None]:
        """A breaker permit if the candidate may be called now, else (None, why not).

        Cheap read-only checks come first; the two checks with side effects (taking a
        breaker probe slot, taking provider capacity) come last, so a skip leaves nothing
        behind."""
        now = self._clock()
        if self._unavailable.get((spec.provider, candidate.model), 0.0) > now:
            return None, "model_unavailable"
        if run.context_floor is not None and not (
            spec.context_window is not None and spec.context_window > run.context_floor
        ):
            return None, "context_window"
        if not deadline.allows(now, self._cfg.min_attempt_s):
            run.deadline_hit = True
            return None, "deadline"
        if self._pause is not None:
            wait = await self._pause.paused_for(spec.provider, spec.provider_model)
            if wait > 0:
                run.waits.append(wait)
                return None, "paused"

        breaker = self._breakers.get(spec.provider, candidate.model)
        permit = breaker.acquire()
        if permit is None:
            if breaker.state is BreakerState.OPEN:
                run.waits.append(breaker.retry_after_s())
            return None, "breaker_open"
        wait = await self._take_provider_capacity(ctx, spec)
        if wait > 0:
            breaker.release(permit)
            run.waits.append(wait)
            return None, "provider_capacity"
        return permit, None

    async def _take_provider_capacity(self, ctx: RequestContext, spec: ModelSpec) -> float:
        """FR-4: traffic is admitted against the capacity of the provider that will serve
        it, so a primary outage can't stampede a small free-tier fallback. Only providers
        with configured rpm/tpm are limited. 0 = admitted."""
        provider = self._config.providers[spec.provider]
        try:
            if provider.rpm is not None:
                wait = await self._store.take(
                    f"prpm:{spec.provider}",
                    capacity=provider.rpm,
                    refill_per_s=provider.rpm / 60,
                    amount=1,
                )
                if wait > 0:
                    return wait
            if provider.tpm is not None:
                needed = prompt_tokens(ctx.request) + output_cap(ctx.request, spec)
                wait = await self._store.take(
                    f"ptpm:{spec.provider}",
                    capacity=provider.tpm,
                    refill_per_s=provider.tpm / 60,
                    amount=min(needed, provider.tpm),
                )
                if wait > 0:
                    if provider.rpm is not None:
                        await self._store.refund(
                            f"prpm:{spec.provider}", capacity=provider.rpm, amount=1
                        )
                    return wait
        except StoreUnavailable:
            # These limits only keep Relay under the provider's own 429s; the pause state
            # still catches those. Not worth refusing traffic over.
            return 0.0
        return 0.0

    async def _call(
        self,
        ctx: RequestContext,
        candidate: Candidate,
        spec: ModelSpec,
        hop: int,
        permit: Permit,
        run: _Run,
        deadline: Deadline,
    ) -> ChatResponse | PrimedStream | None:
        """Call one model, retrying it at most once. None = it failed (see ``run``)."""
        adapter = self._providers[spec.provider]
        strong = self._config.routes[ctx.route].strong_model
        # Priced per attempt: the reservation is settled against the model that served.
        ctx.pricing = Pricing(
            model=spec,
            strong=self._config.models[strong] if strong else None,
            price_table_version=self._config.price_table_version,
        )
        # The provider gets its own model ID and always an output cap: Anthropic requires
        # one, and the reservation was sized against it, so the cap reserved is the cap sent.
        upstream_request = ctx.request.model_copy(
            update={
                "model": spec.provider_model,
                "max_tokens": ctx.request.max_tokens or spec.default_max_tokens,
            }
        )
        self._budget.record_attempt(spec.provider)

        retry = 0
        while True:
            started = self._clock()
            try:
                result = await self._invoke(ctx, adapter, upstream_request, candidate, deadline)
            except ProviderError as e:
                latency = self._clock() - started
                self._breakers.record(permit, e.error_class)
                if self._pause is not None:
                    await self._pause.record(spec.provider, spec.provider_model, e)
                ctx.attempts.append(
                    ProviderAttempt(
                        spec.provider, candidate.model, latency, e.error_class, hop, retry
                    )
                )
                ATTEMPTS.labels(spec.provider, candidate.model, e.error_class.value).inc()
                run.last_error = e
                run.left_because = e.error_class.value
                if e.error_class is ErrorClass.MODEL_NOT_FOUND:
                    self._unavailable[(spec.provider, candidate.model)] = (
                        self._clock() + self._cfg.model_unavailable_ttl_s
                    )
                    MODEL_UNAVAILABLE.labels(spec.provider, candidate.model).inc()

                if retry == 0 and e.error_class in RETRY_ONCE:
                    next_permit = await self._retry_permit(spec, candidate, deadline)
                    if next_permit is not None:
                        permit = next_permit
                        retry = 1
                        continue
                if not deadline.allows(self._clock(), self._cfg.min_attempt_s):
                    run.deadline_hit = True
                if e.error_class in CONTEXT_FALLBACK:
                    # Only a model with a larger *known* window can take this prompt.
                    run.context_floor = float(spec.context_window or math.inf)
                elif not can_fall_back(e.error_class):
                    run.stop = True
                return None

            latency = self._clock() - started
            ctx.attempts.append(
                ProviderAttempt(spec.provider, candidate.model, latency, None, hop, retry)
            )
            ATTEMPTS.labels(spec.provider, candidate.model, "ok").inc()
            if isinstance(result, PrimedStream):
                # The breaker hears about a stream when it *ends* (ProviderStage.finalize):
                # a stream that breaks mid-way is a failure, not the success priming saw.
                ctx.breaker_permit = permit
            else:
                self._breakers.record(permit, None)
                ctx.record_usage(result.usage)
                # Kept for the cache write-back (a stream collects its pieces as it forwards).
                ctx.reply_parts = [result.choices[0].message.content or ""]
                ctx.finish_reason = result.choices[0].finish_reason
            return result

    async def _invoke(
        self,
        ctx: RequestContext,
        adapter: ProviderAdapter,
        upstream_request: ChatRequest,
        candidate: Candidate,
        deadline: Deadline,
    ) -> ChatResponse | PrimedStream:
        now = self._clock()
        if ctx.stream:
            wait = min(
                ctx.tenant.deadlines.first_token_s,
                self._first_token_ceiling_s,
                deadline.remaining(now),
            )
            # Only up to the first content chunk. The rest is pulled by the response body
            # after the headers are committed (see relay/pipeline/streaming.py).
            return await prime(
                adapter.stream(upstream_request), timeout_s=wait, relay_model=candidate.model
            )
        try:
            async with asyncio.timeout(deadline.remaining(now)):
                response = await adapter.complete(upstream_request)
        except TimeoutError:
            raise ProviderError(ErrorClass.TIMEOUT, "the request deadline was reached") from None
        # Clients see Relay names only. Leaking the upstream ID would let clients depend on it.
        return response.model_copy(update={"model": candidate.model})

    async def _retry_permit(
        self, spec: ModelSpec, candidate: Candidate, deadline: Deadline
    ) -> Permit | None:
        """Wait the backoff and return a permit for the one retry, or None if the retry
        budget, the deadline or the breaker says no."""
        delay = full_jitter(
            0, base_s=self._cfg.backoff_base_s, cap_s=self._cfg.backoff_cap_s, rng=self._rng
        )
        if not deadline.allows(self._clock(), delay + self._cfg.min_attempt_s):
            return None
        if not self._budget.try_spend(spec.provider):
            return None
        RETRIES.labels(provider=spec.provider).inc()
        await self._sleep(delay)
        # The first failure may have been the one that opened the breaker.
        return self._breakers.get(spec.provider, candidate.model).acquire()

    # ------------------------------------------------------------------ giving up
    def _give_up(self, ctx: RequestContext, candidates: list[Candidate], run: _Run) -> RelayError:
        last = run.last_error
        if run.stop and last is not None:
            # The request itself was the problem (bad_request, content_filtered): another
            # model wouldn't help, so the client gets that error, as before Phase 7.
            return RelayError.from_provider(last)
        if run.deadline_hit:
            return RelayError(
                504,
                ErrorClass.DEADLINE_EXCEEDED,
                "the request deadline was reached before any provider could answer",
            )
        retry_after = min(run.waits) if run.waits else None
        called = [a for a in ctx.attempts if a.skipped_reason is None]
        skips = {a.skipped_reason for a in ctx.attempts if a.skipped_reason is not None}
        if not called and skips == {"paused"}:
            # Every candidate is rate limited: nothing failed, it just isn't time yet. The
            # client may retry after Retry-After (CLIENT_RETRYABLE), exactly as in Phase 6.
            return RelayError(
                429,
                ErrorClass.PROVIDER_PAUSED,
                "the provider is rate limiting this model; retry later",
                retry_after_s=retry_after,
            )
        if len(candidates) == 1 and last is not None:
            # No fallback was configured for this request (no graph, or an explicit model):
            # the provider's own error says most about what happened.
            return RelayError.from_provider(last)
        return RelayError(
            503,
            ErrorClass.NO_USABLE_FALLBACK,
            f"no usable provider: {len(candidates)} candidate model(s) failed or were "
            f"unavailable (last: {run.left_because})",
            retry_after_s=retry_after if retry_after is not None else 1.0,
        )
