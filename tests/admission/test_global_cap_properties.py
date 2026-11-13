"""Property-based tests for the two-budget reservation (budget and
admission code require them).

A request holds money against its tenant's daily budget *and* against the all-tenant cap.
hypothesis generates sequences of admissions, settlements and clock jumps, and after every step
checks the rules that must always hold:

1. **Neither cap is ever exceeded:** for every budget, committed + outstanding <= its limit.
2. **The two books agree:** every admitted request holds the same amount in both, so the
   global budget's outstanding and committed equal the sums over the tenants'.
3. **A refusal leaves nothing behind:** a request refused by either budget changes no budget.
4. **Everything is released:** once every request has finished, nothing is outstanding.
"""

import copy
from decimal import Decimal
from typing import Any

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from relay.admission.stages import ReservationStage
from relay.admission.store import MemoryStore
from relay.config.pricing import NANO_PER_USD, Pricing
from relay.config.schema import RelayConfig
from relay.pipeline.context import RequestContext
from relay.pipeline.stage import Continue, Outcome
from relay.routing.base import RouteDecision
from relay.schemas.chat import ChatRequest, Usage

from ..conftest import TEST_CONFIG, yaml_policy  # type: ignore[import-not-found]
from .test_properties import Clock, run

TENANT_USD = Decimal("0.02")
GLOBAL_USD = Decimal("0.03")  # less than the two tenants' budgets together: the cap binds
MODEL = "mock/strong"
HOLD_TTL_S = 61.0  # ReservationStage: max(1, time left to the deadline) + 60 s margin


def _config() -> RelayConfig:
    data = copy.deepcopy(TEST_CONFIG.model_dump(mode="json"))
    template = data["tenants"]["test"]
    for name in ("a", "b"):
        data["tenants"][name] = {
            **template,
            "budget_usd_daily": str(TENANT_USD),
            "budget_mode": "hard",
        }
    data["global_budget_usd_daily"] = str(GLOBAL_USD)
    return RelayConfig.model_validate(data)


class TwoBudgets(RuleBasedStateMachine):
    @initialize()
    def setup(self) -> None:
        self.clock = Clock()
        self.store = MemoryStore(self.clock)
        self.config = _config()
        self.stage = ReservationStage(self.store, self.config, planner=lambda ctx: [MODEL])
        self.open: list[RequestContext] = []
        self.expires: dict[int, float] = {}  # id(ctx) -> when its hold would lapse

    def _ctx(self, tenant: str, max_tokens: int) -> RequestContext:
        request = ChatRequest.model_validate(
            {
                "model": "relay/strong",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": max_tokens,
            }
        )
        ctx = RequestContext(
            request_id="r",
            tenant=self.config.tenants[tenant],
            request=request,
            route="default",
            priority="interactive",
            cohort=None,
            policy_version="t",
            deadline=self.clock() + 60,
            stream=False,
            applied=yaml_policy(self.config),
        )
        ctx.decision = RouteDecision(
            tier="strong", model=MODEL, score=None, reason="alias", policy_version="t"
        )
        return ctx

    def _books(self) -> dict[str, tuple[int, int]]:
        # Reading the store's own books: committed and outstanding per budget key.
        # A hold past its expiry no longer counts, though the store only sweeps it out at the
        # next reserve on that budget: it is already free money to whoever asks next.
        now = self.clock()
        return {
            key: (b.committed, sum(a for a, expires in b.outstanding.values() if expires > now))
            for key, b in self.store._budgets.items()
        }

    @rule(tenant=st.sampled_from(["a", "b"]), max_tokens=st.integers(1, 3000))
    def admit(self, tenant: str, max_tokens: int) -> None:
        before = self._books()
        ctx = self._ctx(tenant, max_tokens)
        result = run(self.stage.process(ctx))
        if isinstance(result, Continue):
            self.open.append(ctx)
            self.expires[id(ctx)] = self.clock() + HOLD_TTL_S
        else:
            assert self._books() == before, "a refused request changed a budget"

    @rule(index=st.integers(0, 50), spent=st.floats(0, 1))
    def finish(self, index: int, spent: float) -> None:
        if not self.open:
            return
        ctx = self.open.pop(index % len(self.open))
        assert ctx.reservation is not None
        if spent > 0:
            # It used a share of what it held (never more than the worst case it reserved).
            tokens = max(1, int(ctx.request.max_tokens or 1) * spent)
            ctx.usage = Usage(
                prompt_tokens=1, completion_tokens=int(tokens), total_tokens=1 + int(tokens)
            )
            ctx.pricing = Pricing(
                model=self.config.models[MODEL], strong=None, price_table_version="t"
            )
        run(self.stage.finalize(ctx, Outcome("success" if spent > 0 else "error")))

    @rule(index=st.integers(0, 50))
    def a_replica_dies(self, index: int) -> None:
        """A request that never finishes (its replica crashed): only the hold's TTL frees it."""
        if self.open:
            self.open.pop(index % len(self.open))

    @rule(seconds=st.floats(0, 5000))
    def time_passes(self, seconds: float) -> None:
        # A running request finishes inside its hold's lifetime (the deadline guarantees it, and
        # a stream's hold is far longer), so time may not jump past one of those. Holds of
        # crashed requests are free to lapse, which is what the TTL is for.
        latest = min((self.expires[id(c)] for c in self.open), default=float("inf"))
        self.clock.t += max(0.0, min(seconds, latest - self.clock.t - 1))

    @invariant()
    def neither_cap_is_exceeded(self) -> None:
        limits = {"global": int(GLOBAL_USD * NANO_PER_USD)}
        for key, (committed, outstanding) in self._books().items():
            limit = (
                limits["global"] if key.startswith("global:") else int(TENANT_USD * NANO_PER_USD)
            )
            assert committed + outstanding <= limit, key

    @invariant()
    def the_two_books_agree(self) -> None:
        books = self._books()
        tenant = [v for k, v in books.items() if not k.startswith("global:")]
        glob = [v for k, v in books.items() if k.startswith("global:")]
        if not glob:
            assert not any(c or o for c, o in tenant)
            return
        # Holds expire together (same ttl) and settle together, so the sums match exactly.
        assert sum(c for c, _ in tenant) == glob[0][0]
        assert sum(o for _, o in tenant) == glob[0][1]

    def teardown(self) -> None:
        for ctx in list(self.open):
            run(self.stage.finalize(ctx, Outcome("error")))
        self.clock.t += 10 * HOLD_TTL_S  # whatever crashed requests still hold has lapsed
        self._books()  # (reads only; expiry is by time, not by a sweep)
        for key, (_, outstanding) in self._books().items():
            assert outstanding == 0, f"{key} still holds money after every request finished"


TestTwoBudgets: Any = TwoBudgets.TestCase
TestTwoBudgets.settings = settings(max_examples=60, stateful_step_count=40, deadline=None)
