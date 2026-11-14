"""The demo's chaos panel: break a simulated provider and watch what happens.

Two clients send the same small requests at the same simulator while it is being broken:
**Relay** (which owns retries and fallback) and **a naive retry client** (five tries with
backoff, then one fallback: relay/demo/naive.py). The panel keeps the last 30 seconds of both
and reports success rate and latency, so the difference is visible live.

It only ever touches the in-process *simulator*, never a real provider, and it is bounded on
every side: an incident lasts at most 60 s, traffic is a few requests a second, at most a fixed
number are in flight, and the whole loop stops by itself two minutes after the last time anyone
asked for the numbers.
"""

import asyncio
import contextlib
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from relay.demo.naive import NAIVE_BACKOFF_S, naive_retry
from relay.schemas.chat import ChatRequest
from relay.schemas.errors import ProviderError
from relay.simulator.engine import Simulator
from relay.simulator.scenario import CHEAP, STRONG, ErrorMix

__all__ = ["CHEAP", "MAX_DURATION_S", "STRONG", "ChaosPanel"]

MAX_DURATION_S = 60.0
ACTIONS = ("kill", "slow", "storm", "clear")
SLOW_FACTOR = 6.0


class ChaosPanel:
    def __init__(
        self,
        simulator: Simulator,
        send_relay: Callable[[], Awaitable[tuple[bool, float]]],
        *,
        clock: Callable[[], float] = time.monotonic,
        rate_hz: float = 5.0,  # all requests, both clients together
        window_s: float = 30.0,
        idle_stop_s: float = 120.0,
        max_in_flight: int = 40,
        backoff_s: tuple[float, ...] = NAIVE_BACKOFF_S,
    ) -> None:
        self._simulator = simulator
        self._send_relay = send_relay
        self._clock = clock
        self._period = 2.0 / rate_hz  # a tick sends one request per client
        self._window_s = window_s
        self._idle_stop_s = idle_stop_s
        self._backoff_s = backoff_s
        self._max_in_flight = max_in_flight
        self._samples: dict[str, deque[tuple[float, bool, float]]] = {
            "relay": deque(),
            "naive": deque(),
        }
        self._last_poll = clock()
        self._task: asyncio.Task[None] | None = None
        self._inflight: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------ the controls
    def apply(self, action: str, duration_s: float) -> None:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; one of {', '.join(ACTIONS)}")
        if action == "clear":
            self._simulator.clear_incidents()
            return
        duration = max(1.0, min(float(duration_s), MAX_DURATION_S))
        errors: ErrorMix | None = None
        multiplier = 1.0
        if action == "kill":
            errors = ErrorMix(server_error=1.0)
        elif action == "storm":
            errors = ErrorMix(rate_limited=0.8, retry_after_s=2.0)
        else:  # slow
            multiplier = SLOW_FACTOR
        # The primary only: the fallback model stays healthy, which is the point of a fallback.
        self._simulator.inject(
            duration_s=duration, models=(STRONG,), errors=errors, latency_multiplier=multiplier
        )

    # ------------------------------------------------------------------ the numbers
    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def stats(self) -> dict[str, Any]:
        """The last window of both clients. Asking also keeps the traffic going, and starts it
        if it is not running: nothing sends requests while nobody is looking."""
        self._last_poll = self._clock()
        if not self.running:
            self._task = asyncio.get_running_loop().create_task(self._run())
        return {name: self._summary(name) for name in self._samples}

    def _record(self, client: str, ok: bool, latency_s: float) -> None:
        self._samples[client].append((self._clock(), ok, latency_s))

    def _summary(self, client: str) -> dict[str, Any]:
        cutoff = self._clock() - self._window_s
        samples = self._samples[client]
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        n = len(samples)
        latencies = sorted(s[2] for s in samples)
        return {
            "n": n,
            "success_rate": (sum(1 for s in samples if s[1]) / n) if n else None,
            "p50_s": _quantile(latencies, 0.50),
            "p95_s": _quantile(latencies, 0.95),
        }

    # ------------------------------------------------------------------ the traffic
    async def _run(self) -> None:
        try:
            while self._clock() - self._last_poll < self._idle_stop_s:
                self._spawn(self._one_relay)
                self._spawn(self._one_naive)
                await asyncio.sleep(self._period)
        except asyncio.CancelledError:
            raise
        finally:
            for task in list(self._inflight):
                task.cancel()

    def _spawn(self, make: Callable[[], Coroutine[Any, Any, None]]) -> None:
        if len(self._inflight) >= self._max_in_flight:
            # At the cap: skip this request rather than queue behind slow ones. A queue here
            # would grow without bound while the primary is down and the naive client waits.
            return
        task = asyncio.get_running_loop().create_task(make())
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _one_relay(self) -> None:
        try:
            ok, latency = await self._send_relay()
        except Exception:
            ok, latency = False, 0.0
        self._record("relay", ok, latency)

    async def _one_naive(self) -> None:
        started = self._clock()

        async def attempt(model: str) -> tuple[bool, None]:
            request = ChatRequest.model_validate(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 20,
                }
            )
            try:
                await self._simulator.open(request).complete()
            except ProviderError:
                return False, None
            return True, None

        ok, _, _ = await naive_retry(
            lambda: attempt(STRONG), lambda: attempt(CHEAP), backoff_s=self._backoff_s
        )
        self._record("naive", ok, self._clock() - started)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        tasks = list(self._inflight)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1)]
