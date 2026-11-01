# Load results, M1(a): passthrough, every request traced (OTLP, 100% sampled) (simulator upstream)

Constant arrival rate for 30s per point. k6 (pinned image) -> nginx -> Relay replicas (2 vCPU and 2 worker processes each) -> simulator (20 ms first token). Docker host CPUs: 12. Admission state in Redis. Log rows are kept in memory (no Postgres); no cache, so this is passthrough. Overhead = through-Relay minus direct at the same percentile, so it includes nginx and Redis. A row where achieved < offered, or failures > 0, is a saturated gateway: its latencies measure queueing, not overhead.

| replicas | offered | achieved | dropped | p50 overhead | p95 overhead | p99 overhead | failed |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 48 | 0 | 14.1 ms | 65.5 ms | 825.4 ms | 0.13% |
