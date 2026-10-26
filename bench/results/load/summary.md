# Load results (simulator upstream)

Constant arrival rate for 30s per point. k6 (pinned image) -> nginx -> Relay replicas (2 vCPU and 2 worker processes each) -> simulator (20 ms first token). Docker host CPUs: 12. Admission state in Redis, log rows kept in memory (no Postgres). **M1(a): passthrough, no cache or router yet.** Overhead = through-Relay minus direct at the same percentile, so it includes nginx and Redis. A row where achieved < offered, or failures > 0, is a saturated gateway: its latencies measure queueing, not overhead.

| replicas | offered | achieved | dropped | p50 overhead | p95 overhead | p99 overhead | failed |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 50 | 0 | 21.2 ms | 83.6 ms | 123.5 ms | 0.00% |
| 1 | 100 | 100 | 0 | 63.7 ms | 824.1 ms | 1252.3 ms | 0.00% |
| 1 | 200 | 113 | 1493 | 7075.7 ms | 16823.0 ms | 18227.5 ms | 2.53% |
| 2 | 50 | 50 | 0 | 9.5 ms | 20.6 ms | 21.2 ms | 0.00% |
| 2 | 100 | 100 | 0 | 27.2 ms | 503.4 ms | 1235.7 ms | 0.00% |
| 2 | 200 | 166 | 587 | 2791.0 ms | 4138.0 ms | 5594.2 ms | 0.00% |
