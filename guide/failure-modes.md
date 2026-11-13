# Failure modes

What breaks, how you will see it, and what Relay does about it. The behaviours below are implemented and tested; the chaos scenarios in `bench/chaos/` inject most of them into the simulator and compare Relay with a naive retrying client (numbers in [results.md](results.md)).

A rule that runs through all of it: **Relay would rather refuse a request than quietly give a worse answer or spend money it should not.** Where it can serve safely it keeps serving; where it cannot, it says so with a status code and a message.

## A model provider fails

| What happens upstream | What Relay does | What the caller sees |
|---|---|---|
| One slow or failed call (`overloaded`, `server_error`) | Retries once, only if the request's deadline and the gateway's retry budget allow. The retry budget caps retries at a share of traffic, so a provider outage cannot turn into a retry storm | Usually nothing: a normal answer |
| Repeated failures | The model's circuit breaker opens, and requests skip it without calling it. It later lets a few probes through before closing | Answers from a fallback |
| The provider says "slow down" (`429`) | Pauses that scope for everyone, on every gateway (through Redis), for as long as the provider asked | Answers from a fallback, or `429 provider_paused` with `Retry-After` if there is none |
| A model is retired (`model_not_found`) | Marks it unavailable for 5 minutes and raises an alert | Answers from a fallback |
| A request too long for the model (`context_length_exceeded`) | Falls back only to a model with a *larger known* context window | The answer, or a clear 400 |
| Everything is down | Gives up at the request's deadline, not later | `503 no_usable_fallback`, or `504 deadline_exceeded` |

**Fallback is not always allowed.** A fallback model might be worse. A `strict` tenant (the default) may only fall back along edges marked `safe`; with none, it gets `503 no_usable_fallback` and a `Retry-After`, instead of a silently degraded answer. An `allow_degraded` tenant may use `undecided` edges, and every such response carries `X-Relay-Degraded: true`. See [configuration.md](configuration.md).

**Streams.** Relay commits to a stream at its first content chunk. A failure *before* that point falls back like any other request. A failure *after* it can no longer change the model (the client already has part of an answer), so Relay ends the stream with an error event and does **not** send `[DONE]`: an incomplete answer must never look complete. A stream that goes quiet for longer than `RELAY_STREAM_IDLE_TIMEOUT_S` is ended the same way. When a client disconnects, the upstream call is closed and the money held for it is released.

## A dependency fails

| Dependency | Behaviour |
|---|---|
| **Redis** is down | The exact-match cache is skipped. Rate limits and provider pauses fall back to conservative in-process limits (the configured limit divided by the replica count). **Budgets fail closed:** `hard`-budget tenants get `503 budget_unavailable` until Redis returns, because overspending is worse than being unavailable. The gateway stays ready |
| **Postgres** is down | The gateway keeps serving. Request logs queue and, if the queue fills, the oldest are dropped and counted (`relay_log_dropped_total`): logging never blocks a request. Known API keys keep working from a short-lived in-memory cache (an unknown key gets `503 auth_unavailable`, not a false 401). The semantic cache is off, evidence sampling pauses, and the last policy stays active |
| The **embedding model** cannot load | The semantic cache is off (exact matching still works) and the router sends everything to the strong model. The gateway starts regardless |
| The **router file** is missing or corrupt | Everything goes to the strong model: the quality-safe default |
| The **evidence engine** or its judge is down | The last known-good policy stays active; nothing is promoted without evidence |
| **No provider key** is set | That provider's models are simply unavailable. Relay starts, and the other models work |

The gateway's readiness check (`/readyz`) stays `200` through Redis and Postgres outages and says what is degraded in its body, because failing readiness would make a load balancer remove every gateway at once, which is worse than serving degraded.

## Too much traffic

| Situation | Response |
|---|---|
| A tenant passes its per-minute request or token limit | `429 tenant_rate_limited`, with `Retry-After` and `x-should-retry: true` |
| One client address passes `per_ip_rpm` | `429 client_rate_limited` |
| Too many requests are waiting for a slot | `429 queue_full` or `queue_timeout`. Interactive requests are served before batch ones, and batch is never starved |
| A tenant's daily budget is spent (`hard`) | `429 budget_exceeded`, resetting at 00:00 UTC. A `soft` budget serves and counts |
| The all-tenant `global_budget_usd_daily` is reached | The same `429 budget_exceeded`, naming the whole-gateway limit |
| A body over `RELAY_MAX_BODY_BYTES` | `413 request_too_large`, before the body is read |

Money is held before a request runs, sized for the most expensive model the request could fall back to, and settled to what it actually cost. A hold that is never settled (the gateway died) expires by itself and the money returns to the budget.

## Bad requests and bad keys

An unknown or revoked key is `401 invalid_api_key`, checked before the request body is looked at (so an unauthenticated caller learns nothing about the API). Unsupported parameters (tools, `n > 1`, and similar) are refused by name with a `400`, not silently ignored: dropping them would hand a client a plain-text answer when it expected a tool call.

Every error uses the OpenAI error envelope, so SDKs raise their usual typed exceptions. Relay sets `x-should-retry: false` on errors it has already retried (it is the single owner of retries, so a client SDK retrying too would multiply the load), and `true` only where waiting helps.

## The key pepper is lost

API keys are stored only as hashes made with the pepper. If the data volume is lost while the database survives, every key would silently stop matching. Relay detects this on start (a freshly generated pepper next to existing keys) and refuses to start, naming the cause. Restore the volume, or set `RELAY_KEY_PEPPER` to the original value.

## Quality drifts without any error

The hardest failure: a provider keeps answering `200` but the answers get worse. Nothing in the error rate changes. Detecting it takes measurement, which is what the evidence engine is for: it samples traffic, has a judge compare cheap and strong answers, and corrects the judge's mistakes with a small set of human labels, producing a confidence interval per comparison. A comparison whose lower bound falls under the floor turns `unsafe`, which closes the fallback edge or rolls back a canary. See [design.md](design.md) and the evidence results in [results.md](results.md).
