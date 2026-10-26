"""Chaos metrics: pure functions over per-request results (PRD E-C, M2, FR-4).

Definitions, fixed in ADR 0012 so a number always means the same thing:

- **Success (M2):** the client received a complete 2xx. A stream must end with ``[DONE]`` and
  carry no error event. Relay's own 429/503 and mid-stream failures are failures. A response
  served through an un-evidenced edge counts as a success and is also counted as *degraded*.
- **Retry amplification:** upstream calls ÷ client requests, measured at the simulator (where
  the load lands), not by trusting the client's own count.
- **Burstiness:** inside a time window, the busiest 100 ms bucket of upstream arrivals at the
  primary ÷ what the *offered* load would put in an average bucket. Around 1 means the
  upstream never saw more than the clients offered; a large number means retries piled up
  into a spike. Relative to the offered rate, not to the window's own mean, because a breaker
  that quiets the upstream would otherwise make a single probe look like a huge spike. This is
  FR-4's "no synchronized retry spike", made a number.
"""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    ok: bool
    latency_s: float
    status: int = 0
    degraded: bool = False
    # A stream that ended with neither [DONE] nor an error event: a silently cut answer.
    silent_truncation: bool = False
    stream: bool = False
    # Which upstream answered and its ground-truth index, for correctness lookups.
    served_by: str | None = None
    index: int | None = None
    started_s: float = 0.0  # seconds since the run began


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Empty input gives nan, not a made-up zero."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(results: Sequence[Result]) -> dict[str, float]:
    n = len(results)
    ok = [r for r in results if r.ok]
    latencies = [r.latency_s for r in ok]
    return {
        "requests": n,
        "success_rate": len(ok) / n if n else math.nan,
        "degraded": sum(r.degraded for r in ok),
        "silent_truncations": sum(r.silent_truncation for r in results),
        "p50_s": percentile(latencies, 50),
        "p95_s": percentile(latencies, 95),
        "p99_s": percentile(latencies, 99),
    }


def retry_amplification(upstream_calls: int, client_requests: int) -> float:
    return upstream_calls / client_requests if client_requests else math.nan


def burstiness(
    arrivals: Mapping[int, int], *, start_bucket: int, end_bucket: int, offered_per_bucket: float
) -> float:
    """Busiest upstream bucket in [start_bucket, end_bucket) ÷ the offered load per bucket."""
    counts = [arrivals.get(b, 0) for b in range(start_bucket, end_bucket)]
    if not counts or offered_per_bucket <= 0:
        return math.nan
    return max(counts) / offered_per_bucket


def mean_and_spread(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation over runs (§11.1: chaos numbers are reported as
    mean ± spread over >= 5 runs, because event interleaving is not deterministic)."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return math.nan, math.nan
    return statistics.fmean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0


def wrong_rate(flags: Sequence[bool]) -> float:
    """Share of served answers whose hidden ground truth was *incorrect*."""
    return sum(1 for correct in flags if not correct) / len(flags) if flags else math.nan
