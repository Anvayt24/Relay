# Configuration

Relay has two kinds of configuration, kept apart on purpose:

- **Environment variables** (`RELAY_*`, and the provider keys) say how the process runs and where its dependencies are. Docker Compose reads them from `.env`.
- **A YAML file** (`relay.yaml`) says what the gateway does: which providers and models exist, what they cost, which routes and tenants there are, and the limits each tenant gets. The default is `relay/config/relay.yaml`. To use your own, point `RELAY_CONFIG_PATH` at it (mount it into the container).

The YAML is validated when Relay starts. A mistake (a typo in a field name, a model that names an unknown provider, a cyclic fallback graph) stops the start with a message that names the problem, instead of surfacing at 3 a.m. as a wrong answer. Secrets never go in the YAML: it holds only the *names* of the environment variables that hold keys.

## The YAML file

```yaml
price_table_version: "2026-11-13"   # a label recorded with every cost, so old costs stay explainable

providers:
  groq: {kind: openai, base_url: "https://api.groq.com/openai/v1", api_key_env: GROQ_API_KEY}
  anthropic: {kind: anthropic, base_url: "https://api.anthropic.com", api_key_env: ANTHROPIC_API_KEY}

models:
  groq/gpt-oss-20b:  {provider: groq, provider_model: "openai/gpt-oss-20b",  tier: cheap,  price_in: 0.075, price_out: 0.30}
  groq/gpt-oss-120b: {provider: groq, provider_model: "openai/gpt-oss-120b", tier: strong, price_in: 0.15,  price_out: 0.60}

routes:
  default:
    aliases: {relay/cheap: groq/gpt-oss-20b, relay/strong: groq/gpt-oss-120b, relay/auto: groq/gpt-oss-120b}
    fallback_graph:
      groq/gpt-oss-120b: [{to: groq/gpt-oss-20b, seed_state: safe}]

tenants:
  team-a:
    route: default
    limits: {rpm: 600, tpm: 2000000, concurrency: 32}
    budget_usd_daily: 5.00
    allowed_providers: [groq]
```

Clients only ever see the names on the left (`relay/cheap`, `groq/gpt-oss-20b`). The upstream id (`provider_model`) is an implementation detail, so a provider renaming a model is a one-line change here.

**Prices are list prices**, in USD per million tokens, even for models you use on a free tier. That is what makes "cost saved" mean something: a free tier would otherwise read as $0 saved.

### providers

| Field | Default | Meaning |
|---|---|---|
| `kind` | required | `openai` (OpenAI-compatible: Groq, OpenRouter, Gemini's compatibility endpoint, and many others), `anthropic`, or `mock` (the in-process simulator) |
| `base_url` | | the API's base address |
| `api_key_env` | | the environment variable holding the key |
| `limit_scope` | `model` | where the provider enforces rate limits: per `model`, or per `account` across all models |
| `max_tokens_param` | `max_tokens` | the request field this provider reads the output cap from (Groq wants `max_completion_tokens`) |
| `rpm`, `tpm` | none | the provider's own limits, which Relay reserves against so it stays under them |

### models

| Field | Default | Meaning |
|---|---|---|
| `provider`, `provider_model` | required | which provider, and the upstream model id |
| `tier` | required | `cheap` or `strong`: what the router and the evidence engine compare |
| `price_in`, `price_out` | required | list USD per million input / output tokens |
| `price_cached_in`, `price_cache_write` | the input price | prices for cached input and cache writes, if the provider has them |
| `billing` | `list` | `free` means a free tier: billed $0, but the list cost is still computed and logged |
| `default_max_tokens` | 1024 | the output cap sent when the client sets none (Relay always sends one, because budgets are reserved against it) |
| `context_window` | unknown | tokens the model accepts; a context-length error may fall back only to a model with a larger known window |
| `thinking` | `default` | `off` turns off a model's hidden reasoning (the Anthropic API only), for models that think by default and bill it as output |

### routes

A route is a set of model aliases, and what to do when a model fails. Tenants are attached to a route.

| Field | Default | Meaning |
|---|---|---|
| `aliases` | required | client-facing name to model. `relay/auto` is the router's name |
| `fallback_graph` | none | for each model, the models it may fall back to |
| `max_fallback_hops` | 2 | how many fallbacks one request may use |
| `cache` | | `semantic_threshold` (a measured number; do not lower it without measuring), `lexical_guard`, `embed_timeout_s` |
| `router` | none | a trained router for `relay/auto`: `artifact` (a JSON file), `threshold`. Without one, `relay/auto` goes where the alias says |
| `evidence` | none | the evidence engine for the route: see below |

**Fallback edges** have a state. `seed_state` is `safe`, `undecided` (the default) or `unsafe`. A tenant with `fallback_policy: strict` may only use `safe` edges; `allow_degraded` may also use `undecided` ones, and the response says `X-Relay-Degraded: true` when it did. With no evidence yet nothing is known to be safe, so a new strict route returns an error rather than silently falling back to a worse model. Mark an edge `safe` yourself when you know the target is as good, or turn on the evidence engine to measure it.

**The evidence engine** (`evidence:` on a route) samples a share of traffic, asks a *judge* model whether the cheap answer was materially worse than the strong one, and turns those verdicts into per-comparison *route cards*. It is off unless a route has this section **and** a tenant sets `evidence_sampling: on`. `judge` names the judge model, `judge_prompt_version` its rubric (`v2` also records the judge's one-line reason), `judge_temperature` its temperature (`null` sends none: required by Claude Sonnet 5.5), `judge_max_tokens` its output cap (16 fits a one-word verdict; a judge that writes a reason needs a few hundred), `sample_rate` and `sample_rate_decided` how much traffic is sampled, and `shadow_budget_usd_per_day` caps what sampling may spend. A judge should be one whose agreement with human labels you have measured: the repository's `evals/judge/public` shows how.

The rest of the section sets the statistics, and its defaults are the pre-registered ones:

| Field | Default | Meaning |
|---|---|---|
| `floor` | 0.90 | a comparison is *safe* when the lower bound of its win-or-tie rate is at least this |
| `canary_sample_rate` | 0.25 | how much of a canary's cheap traffic is sampled (a canary needs a decision quickly) |
| `look_every`, `max_looks`, `alpha` | 500, 4, 0.05 | decisions are made only every `look_every` judged pairs, at most `max_looks` times, with the error rate `alpha` split across them |
| `min_cohort_n` | 200 | cohorts smaller than this are merged into `other` |
| `rectifier_max_age_days` | 30 | human labels older than this make a card `stale` |
| `interval` | `wald` | the confidence-interval method (`wald` or `bootstrap`) |
| `band_lo` | none | also sample the cheap model on strong-routed requests whose router score is in `[band_lo, threshold)`: the only evidence that can support lowering the threshold |
| `sample_fallbacks` | true | also compare each gated fallback target with a healthy primary, gathered before an outage needs it |

### tenants

A tenant is who is calling. It is found from the API key.

| Field | Default | Meaning |
|---|---|---|
| `route` | `default` | the route it uses |
| `limits` | required | `rpm`, `tpm`, `concurrency` (default 8), and optionally `per_ip_rpm`: requests per minute from one client address |
| `request_limits` | none | `max_messages`, `max_tokens`: what one request may ask for. A request over `max_tokens` is refused, and one that sets none gets exactly this many |
| `budget_usd_daily` | required | daily spend ceiling in USD, at billed price, resetting at 00:00 UTC |
| `budget_mode` | `hard` | `hard` refuses requests that would pass the budget; `soft` serves them and counts them |
| `fallback_policy` | `strict` | see fallback edges above |
| `logging` | `metadata_only` | `full` also stores prompt text for evidence, deleted after `retention_days` (default 30) |
| `default_priority` | `interactive` | `interactive` or `batch`; interactive is served first |
| `evidence_sampling` | `off` | whether this tenant's traffic may be shadow-sampled |
| `allowed_providers` | required | the only providers that may ever see this tenant's prompts (primary, fallback, shadow and judge calls alike) |
| `allowed_models` | `*` | restrict to named models |
| `cohort_tags` | none | `X-Relay-Cohort` values kept as given; any other becomes `other` |
| `fallback_on_explicit_model` | false | whether a request naming a provider model directly (not an alias) may also fall back |
| `deadlines` | 15 / 60 / 120 s | `first_token_s` (per attempt), `interactive_s` and `batch_s` (total): fixed when the request arrives, they bound every retry and fallback before the first byte |
| `cache` | off | `enabled`, `semantic`, `ttl_s`, `allow_unset_temperature`, `max_prompt_tokens`. Caching stores model output, so a tenant opts in |

### Whole-gateway ceiling and reliability

`global_budget_usd_daily` (top level, optional) caps billed spend **across all tenants** per day, on top of each tenant's own budget. A request that would pass it is refused, and so is every request while the budget store is unreachable.

`reliability:` tunes retries, deadlines and circuit breakers. The defaults are measured; change them only with a reason.

| Field | Default | Meaning |
|---|---|---|
| `backoff_base_s`, `backoff_cap_s` | 0.2, 2.0 | the wait before a retry, and its ceiling |
| `retry_budget_ratio`, `retry_budget_min`, `retry_budget_window_s` | 0.1, 3, 10 | retries may add at most this share of recent traffic (and always at least `retry_budget_min` per window), so an outage cannot become a retry storm |
| `breaker_window_s`, `breaker_failure_ratio`, `breaker_min_calls` | 30, 0.5, 10 | a model's breaker opens when this share of its calls in the window failed, once it has seen enough calls |
| `breaker_open_s`, `breaker_half_open_probes` | 30, 3 | how long it stays open, then how many probes it lets through before closing |
| `min_attempt_s` | 0.5 | an attempt with less than this left before the deadline is not started |
| `model_unavailable_ttl_s` | 300 | after a `model_not_found`, how long that model is skipped |

## Environment variables

Provider keys: `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (read by name from the `api_key_env` of each provider). Everything else is prefixed `RELAY_`.

### Running the process

| Variable | Default | Meaning |
|---|---|---|
| `RELAY_ENV` | `dev` | `dev`, `test` or `prod`. `prod` turns off the API documentation pages and refuses `RELAY_AUTH_DISABLED` |
| `RELAY_CONFIG_PATH` | the shipped file | your `relay.yaml` |
| `RELAY_LOG_LEVEL`, `RELAY_LOG_FORMAT` | `INFO`, `text` | `json` writes one JSON object per line with `request_id` and `trace_id` |
| `RELAY_DATA_DIR` | unset | a directory that survives restarts; generated secrets are kept here |
| `RELAY_REPLICA_COUNT` | 1 | how many gateways share the limits (needs `RELAY_REDIS_URL` above 1) |
| `RELAY_SHUTDOWN_DRAIN_S` | 10 | how long shutdown waits for open streams to finish |

### Dependencies

| Variable | Default | Meaning |
|---|---|---|
| `RELAY_DATABASE_URL` | unset | Postgres. Without it requests are logged in memory only, and keys cannot be stored |
| `RELAY_REDIS_URL` | unset | Redis, for limits and budgets shared by several gateways. Without it, admission state is in the process (one gateway only) |
| `RELAY_EMBEDDER` | `fastembed` | the model that embeds prompts for the semantic cache and the router: `fastembed`, `hash` (a cheap stand-in for tests) or `off` |

### Secrets and keys

| Variable | Default | Meaning |
|---|---|---|
| `RELAY_KEY_PEPPER` | generated | the secret API-key hashes are computed with. If unset and `RELAY_DATA_DIR` is set, one is generated and kept there |
| `RELAY_BOOTSTRAP_KEY`, `RELAY_BOOTSTRAP_TENANT` | on, `dev` | create and print one API key for this tenant on the first start |
| `RELAY_ADMIN_API`, `RELAY_ADMIN_TOKEN` | off | turn on `/v1/admin/policy` (policy versions, canaries, rollback); the token is generated if unset |
| `RELAY_AUTH_DISABLED`, `RELAY_AUTH_DISABLED_TENANT` | false, `dev` | skip authentication and treat every request as one tenant. Local development only (refused when `RELAY_ENV` is `prod`) |

### Limits and timeouts

| Variable | Default | Meaning |
|---|---|---|
| `RELAY_MAX_BODY_BYTES` | 1048576 | the largest request body accepted; above it the answer is 413 |
| `RELAY_TRUSTED_PROXIES` | none | addresses or CIDR blocks of the reverse proxies in front of Relay; only their `X-Forwarded-For` is believed |
| `RELAY_REQUEST_TIMEOUT_S`, `RELAY_CONNECT_TIMEOUT_S` | 60, 5 | upstream call timeouts |
| `RELAY_FIRST_TOKEN_TIMEOUT_S`, `RELAY_STREAM_IDLE_TIMEOUT_S` | 15, 30 | how long a stream may take to start, and how long it may go quiet |
| `RELAY_QUEUE_MAX_WAIT_INTERACTIVE_S`, `RELAY_QUEUE_MAX_WAIT_BATCH_S`, `RELAY_QUEUE_MAX_DEPTH` | 2, 30, 100 | how long a request may wait for a slot, and how many may wait |

### Observability

| Variable | Default | Meaning |
|---|---|---|
| `RELAY_METRICS_PORT` | unset | serve `/metrics` on this port instead of the API's; keep it off the network and let Prometheus scrape it |
| `RELAY_OTEL_ENDPOINT`, `RELAY_OTEL_SAMPLE_RATIO` | unset, 1.0 | send traces to an OTLP collector, and how many to keep |
| `RELAY_LOG_QUEUE_MAX`, `RELAY_LOG_BATCH_MAX`, `RELAY_LOG_FLUSH_INTERVAL_S`, `RELAY_LOG_SHUTDOWN_TIMEOUT_S` | 10000, 500, 0.5, 5 | the request log's buffering. Logging is never on the request path |

### Evidence and policy

| Variable | Default | Meaning |
|---|---|---|
| `RELAY_CARD_POLL_S`, `RELAY_CARD_MAX_AGE_S` | 5, 600 | how often cards are re-read, and when a card the worker has not refreshed is ignored |
| `RELAY_POLICY_POLL_S` | 5 | how often each gateway re-reads the active policy |
| `RELAY_EMBEDDER_ONNX_THREADS` | 1 | threads for the embedding model |

### The demo page and simulator

| Variable | Default | Meaning |
|---|---|---|
| `RELAY_UI`, `RELAY_WEB_DIR` | on, `web` | serve the demo page at `/` |
| `RELAY_DEMO_TENANT`, `RELAY_DEMO_SIM_TENANT` | unset | the tenants the page's keyless chat answers as: real (free) models, and the simulator |
| `RELAY_DEMO_CHAOS`, `RELAY_DEMO_CHAOS_TENANT` | off, `demo-chaos` | the chaos panel (simulator only; needs `RELAY_SIM_SCENARIO`) |
| `RELAY_SIM_SCENARIO`, `RELAY_SIM_SEED` | unset, 0 | make the `mock` provider a seeded simulator (`ideal`, `realistic`) |

Nothing in the demo group is on unless you set it; a normal deployment does not carry it.
