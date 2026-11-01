# Tracing overhead (Phase 14): every request traced vs not traced

`python -m bench.load --rates 50 --replicas 1 [--tracing] --suffix _tracing_{off,on}_{a,b}`. The same
load stack as M1(a): k6, then nginx, then 1 Relay replica (2 vCPU, 2 workers), then the simulator
(20 ms first token). Tracing on means 100% sampled, `BatchSpanProcessor` exporting over OTLP gRPC to
a collector that discards the spans. Off and on were **alternated twice** (off, on, off, on) on
2026-11-01, so slow periods on the host hit both settings. Overhead is through-Relay minus direct
at the same percentile.

| run | achieved / offered | p50 overhead | p95 overhead | p99 Relay |
|---|---|---|---|---|
| off, round a | 50 / 50 | 17.1 ms | 188.3 ms | 1093.9 ms |
| on, round a | 50 / 50 | 18.8 ms | 297.1 ms | 1224.6 ms |
| off, round b | 49 / 50 | 12.5 ms | 296.4 ms | 1502.1 ms |
| on, round b | 48 / 50 | 14.1 ms | 65.5 ms | 858.3 ms |

**Reading it:**
- **Median cost: about 1.6 ms per request.** The p50 difference is +1.7 ms in round a and +1.6 ms
  in round b, which is consistent.
- **No tail cost can be measured here.** The p95 swings between runs of the *same* setting
  (188 vs 296 ms off) are far larger than any difference between settings, and the fastest p95 of
  all came from a traced run. On this host today the tail measures the machine, not tracing.
- **The host was noisy.** The Phase 8 M1(a) baseline at this point was a p95 of 84 ms through
  Relay, and here it is 218-327 ms untraced. Repeat on a quiet machine before citing a tail number.
- **The first attempt** (also 2 replicas) was discarded: its untraced 2-replica p95 was 2.4 s against
  20 ms in M1, and the traced stack hit a transient `compose up` failure. The harness now retries once.
