# Results

How the current system performs, measured. These are baselines to improve on, not scores against a pass mark. Each section says what was measured and how, and ends with where it has room to get better.

Every number here comes from a command in this repository.

- **`make reproduce`** regenerates the evaluation numbers (cache, router, evidence, judge agreement) from frozen data and fixed seeds, offline, and fails if any number differs from the committed one.
- **`make chaos`** and **`make load`** are timing runs in which a real clock decides the outcome. They cannot be bit-identical, so they are reported as a mean and a spread over several runs.

Unless a line says otherwise, results come from the **provider simulator** or from **public datasets**, not from real provider traffic. A simulator shows how Relay behaves when a provider fails in a stated way. It does not tell you how often a real provider fails.

## Reliability under faults

`make chaos` starts two simulated providers (a primary and a fallback) and Relay, injects a fault into the primary, and sends the same traffic through three clients: **Relay**, a **naive retry client** (five tries with growing waits and no jitter, then one fallback), and a **direct** client (one call, no retry). It reports the mean over **5 runs**.

Success means a complete 2xx response; a stream must end with `[DONE]`. Relay's own refusals and mid-stream failures count as failures.

| Scenario | Client | Success | p95 latency | Upstream calls per request |
|---|---|---|---|---|
| **Headline**: 20% of primary calls return 429, plus one 60 s hard outage (120 s, 20 req/s) | Relay | 99.9% | 0.67 s | 1.08 |
| | Naive retry | 100.0% | 54.35 s | 2.44 |
| | Direct | 39.6% | 0.45 s | 1.00 |
| **Hard outage**: the primary fails every call for 30 s (60 s, 20 req/s) | Relay | 100.0% | 0.70 s | 1.17 |
| | Naive retry | 100.0% | 15.22 s | 2.84 |
| | Direct | 50.3% | 0.47 s | 1.00 |
| **429 storm**: the primary answers 429 to 80% of calls for 30 s | Relay | 100.0% | 0.52 s | 1.04 |
| | Naive retry | 100.0% | 8.95 s | 2.24 |
| | Direct | 60.6% | 0.45 s | 1.00 |

What it shows:

- **Both Relay and the naive client end up succeeding**, because both eventually reach the fallback. The difference is *how*. The naive client waits through its backoff schedule, so a user waits 9 to 54 seconds at p95, and it sends 2.2 to 2.8 times as many upstream calls, which is how a retrying client turns a provider outage into a bigger one. Relay answers in about half a second and adds 4% to 17% extra calls.
- **The direct client** shows what a fault costs with no protection: 40% to 61% success.
- **Relay's few failures.** In the headline scenario 7 of about 12,100 requests failed across the five runs (none in one run, up to 3 in another). The run records only totals, so the cause of those has not been traced.
- **Scope.** Only the three scenarios above were run for 5 runs each. The other four (latency spike, mid-stream drop, silent degradation, cold-start strict) run as compressed smoke tests on every pull request (`make chaos-smoke`), which assert the documented behaviour but are too short to report as numbers. `make chaos` runs all of them. Relay has not been compared with other gateways.
- The runs shared a machine with other work, so absolute latencies would move on another host; the *ratios* between the clients are the result.

**Room to improve:** trace the rare failures at the start of an outage; run all seven scenarios for the full 5 runs on a quiet machine.

## Gateway overhead

`make load` sends constant-rate traffic through nginx and one or two Relay replicas (2 vCPU each) to a simulator with a 20 ms first token, and subtracts the same traffic sent straight to the simulator. This is the passthrough configuration, with the cache and router off.

| Replicas | Offered | Achieved | p50 overhead | p95 overhead | Failed |
|---|---|---|---|---|---|
| 1 | 50 req/s | 50 | 21.2 ms | 83.6 ms | 0% |
| 1 | 100 req/s | 100 | 63.7 ms | 824.1 ms | 0% |
| 1 | 200 req/s | 113 | 7075.7 ms | 16823.0 ms | 2.5% |
| 2 | 50 req/s | 50 | 9.5 ms | 20.6 ms | 0% |
| 2 | 100 req/s | 100 | 27.2 ms | 503.4 ms | 0% |
| 2 | 200 req/s | 166 | 2791.0 ms | 4138.0 ms | 0% |

One replica keeps up to about 100 requests per second; at 200 it stops keeping up, and rows where the achieved rate is below the offered rate measure queueing, not overhead. Two replicas roughly double the headroom. Turning tracing on costs about 1.6 ms at p50.

**Room to improve:** the per-request overhead at 100 requests/s per replica, which is where the p95 climbs.

## Semantic cache

On 2,946 labelled prompt pairs (1,498 to choose the threshold, 1,448 held out), the similarity threshold is the smallest one that reached 99% precision on the tuning half, and everything below is on the held-out half.

| Lexical guard | Threshold | Held-out precision (95% lower bound) | Hit rate |
|---|---|---|---|
| off | 0.9450 | 99.22% (98.00%) | 26.6% |
| **on** (what Relay ships) | **0.8625** | **99.35% (98.53%)** | **42.7%** |

The guard (matching numbers and names) lets the threshold drop from 0.9450 to 0.8625 at the same precision, which is why the hit rate rises from 26.6% to 42.7%. The labels are **constructed**, not judged by humans (same fact in different words is a match; a changed fact is not), and about 43% of the pairs are true paraphrases, so the hit rate reflects how the set was built, not production traffic.

**Room to improve:** human-judged labels, and a set drawn from real traffic.

## Router

A logistic regression over a prompt embedding sends a request to the cheap model when its estimated chance of being as good as the strong model's answer is above a threshold chosen on a validation split.

| World | Share sent cheap | Quality (95% lower bound) |
|---|---|---|
| Simulated, difficulty built into the prompts | 37.2% | 0.986 (0.978) |
| Real prompts (small set, uncalibrated judge) | 3.7% | not meaningful; AUC 0.506, no detectable signal |

On the simulated world the router is far better than random and saves about 33.5% of list-price cost at that quality. The real-prompt run has too few prompts and an unvalidated judge to say anything; it shows the machinery works on real data, not that routing saves money on it. The shipped real-model routes have no router.

**Room to improve:** a larger real prompt set, labelled by a judge that has been checked against humans, which is the only way to learn whether routing helps on real traffic.

## Evidence engine

- **Interval coverage.** 54 simulated cells (number of human labels, number of judged pairs, true win rate, judge quality), 1,000 datasets each. The interval the route cards use (Wald) covers at least 93% in 50 of the 54 cells; the worst cell is **91.0%** (few human labels, true win rate near the floor). The bootstrap alternative is worse (worst 80.3%), so the cards keep Wald. With 1,000 runs a coverage estimate has a standard error of about 0.8 points, so cells within 1 point of 93% are not clearly below it.
- **Detection delay.** For a drop in the win rate from 0.95 to 0.85, at the same false-alarm rate, a rolling fixed threshold detects in a median of **254** judged samples and the evidence monitor in **306**. The fixed threshold is faster here. The evidence monitor's value shows elsewhere.
- **What the correction buys.** With a judge that never notices real losses, a judge-only estimate declares a bad candidate safe **94%** of the time; with the human-label correction it does so **0.1%** of the time. With a judge whose bias drifts after labelling, the corrected estimate's false-safe rate rises to **17.7%**, which is why cards go stale and need fresh labels.
- **Canary decisions.** A bad canary (true rate 0.84) is rolled back in **68%** of simulated runs and never recommended for promotion. A good one (0.97) is recommended in **99%** of runs and never rolled back.

**Room to improve:** interval coverage with few human labels; detection delay.

## The judge

Relay uses a judge only if it agrees with human labels: Cohen's κ of at least 0.6, and it must catch at least 70% of the losses humans labelled. It measured two candidates on 420 public pairs: 300 MT-Bench human judgments and 120 LLMBar pairs, half of them adversarial.

| Judge | Pairs judged | κ [95% CI] | Loss recall [95% CI] | Used? |
|---|---|---|---|---|
| Claude Sonnet 5.5, rubric v2, thinking off | 416 | **0.612** [0.539, 0.688] | **0.764** [0.709, 0.817] | yes |
| Gemini 3.1 Flash-Lite, rubric v1 | 420 | 0.494 [0.405, 0.572] | 0.765 [0.706, 0.821] | no |

Sonnet clears the rule on the point estimate, and its confidence interval for κ includes values under 0.6, so the pass is narrow. Two humans labelling the same MT-Bench pairs agree at only κ = 0.539 (148 pairs), which is about the ceiling a judge can be expected to reach there. This is agreement on **public data**, not on Relay's own traffic.

**Room to improve:** labels from Relay's own traffic; a judge that is both cheaper and as accurate.

## Evidence cost

Evidence cost is the shadow call plus the judge call, priced at list price, as a share of routed spend, measured on 1,000 simulated requests.

| Judge | Sampling 5% (default) | Sampling 1% (once a card is decided) |
|---|---|---|
| Cheap model | 3.8% | 0.9% |
| Claude Sonnet 5.5 (the judge in use), 80 output tokens assumed per judgment | 23.2% | 5.3% |

The judge in use costs real money per judgment, so at the default sampling rate the evidence costs about a quarter of what is routed. Sampling rate lowers it directly, and `shadow_budget_usd_per_day` caps the absolute spend either way.

**Room to improve:** a cheaper judge that has itself been checked against human labels, with the strong judge only auditing; a lower default sampling rate.

## Reproduce

```sh
make reproduce        # the evaluation numbers: offline, a few minutes, fails if any changed
make chaos-smoke      # compressed fault scenarios, asserted on every pull request
make chaos            # the full fault runs (about 100 minutes)
make load             # the load test (needs Docker)
```
