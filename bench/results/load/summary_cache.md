# Load results, M1(b): cache on, embedding on every request (simulator upstream)

Constant arrival rate for 30s per point. k6 (pinned image) -> nginx -> Relay replicas (2 vCPU and 2 worker processes each) -> simulator (20 ms first token). Docker host CPUs: 12. Admission state in Redis. Postgres + pgvector sit behind the cache and every prompt is unique, so each request embeds, queries pgvector, calls the provider and writes the answer back: the heaviest path the cache adds. Overhead = through-Relay minus direct at the same percentile, so it includes nginx and Redis. A row where achieved < offered, or failures > 0, is a saturated gateway: its latencies measure queueing, not overhead.

| replicas | offered | achieved | dropped | p50 overhead | p95 overhead | p99 overhead | failed |
|---|---|---|---|---|---|---|---|
| 1 | 25 | 23 | 5 | 2835.8 ms | 11549.9 ms | 14367.5 ms | 0.40% |
| 1 | 50 | 33 | 160 | 21.4 ms | 18390.8 ms | 20721.2 ms | 0.00% |
