# Self-hosting Relay

Relay runs as a few containers: the gateway, Postgres and Redis. This page covers starting it, using it, keeping it running, and growing it.

## Start

You need Docker with Compose. From a clone of the repository:

```sh
docker compose up -d --build --wait
docker compose logs api        # the first API key is printed here, once
```

The gateway listens on `http://127.0.0.1:8000`. No provider key is needed to try it: out of the box it serves built-in simulator models, so you can send requests immediately.

What happened on that first start:

- **Secrets were generated.** The key pepper (the secret that API-key hashes are computed with) was created and stored in the `relaydata` volume, owner-only. You did not have to invent one.
- **The database was set up.** A one-shot `migrate` container applied the schema before the gateway started.
- **One API key was created** for the `dev` tenant and printed once. Only its hash is stored.

If you lose the key, make another:

```sh
docker compose exec api python -m relay.auth create-key --tenant dev
docker compose exec api python -m relay.auth list
docker compose exec api python -m relay.auth revoke --prefix rk_3fA9xQ2
```

## Use it

Relay speaks the OpenAI chat-completions API, so any OpenAI client works. Point it at Relay:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="rk_...")
reply = client.chat.completions.create(
    model="relay/auto",
    messages=[{"role": "user", "content": "Say hello."}],
)
print(reply.choices[0].message.content)
```

The model names a route provides are `relay/cheap`, `relay/strong` and `relay/auto` (the router chooses). `GET /v1/models` lists what your key may call.

Every response says what was decided, in headers:

| Header | Meaning |
|---|---|
| `X-Relay-Provider`, `X-Relay-Model` | who actually answered (after any fallback) |
| `X-Relay-Cache` | `hit`, `miss` or `bypass` |
| `X-Relay-Router-Score` | the router's estimate that the cheap model is good enough |
| `X-Relay-Fallback-Hops` | 0 when the primary answered |
| `X-Relay-Cost-USD`, `X-Relay-Strong-Cost-USD` | what it cost, and what always-strong would have cost (list price) |
| `X-Relay-Latency-Ms` | time until the response started |
| `X-Request-ID` | the same id is on the request's log lines and trace |

Streaming works (`stream: true`); for a stream the cost arrives in the final chunk, under a top-level `relay` key, when you ask for usage (`stream_options.include_usage`).

## Use real models

Copy `.env.example` to `.env`, add the provider keys you have, and restart the gateway:

```sh
cp .env.example .env
# edit .env: GROQ_API_KEY=..., ANTHROPIC_API_KEY=..., ...
docker compose up -d
```

A model whose provider has no key is simply unavailable; Relay still starts. Which models exist, their prices, and which routes and tenants use them is configuration: see [configuration.md](configuration.md).

## Dashboards and traces

```sh
docker compose --profile observability up -d
```

This adds Prometheus, Grafana and Tempo. Set `RELAY_OTEL_ENDPOINT=http://otel-collector:4317` and `GRAFANA_PASSWORD` in `.env` first. Grafana is at `http://127.0.0.1:3000` with a provisioned "Relay overview" dashboard: traffic, latency, cost saved against always-strong, cache hit rate, per-provider p95, fallbacks and breaker state. Metrics are served on a separate port (`RELAY_METRICS_PORT`) that is not published to the host.

## Reaching it from other machines

By default only `127.0.0.1` can connect. To accept connections from elsewhere set `RELAY_BIND=0.0.0.0` in `.env`, and **put TLS in front** (a reverse proxy such as Caddy or nginx). Then tell Relay which proxy to trust for client addresses, so per-address limits count real clients and not the proxy:

```
RELAY_TRUSTED_PROXIES=10.0.0.0/8
```

Relay serves its API documentation pages only in development; with `RELAY_ENV=prod` (the compose default) they are off.

## Upgrade

```sh
git pull
docker compose up -d --build
```

The schema migration runs first, automatically, and only applies what is new. Take a backup before upgrading a deployment that matters.

## Back up

Two things hold state, and you need both:

```sh
docker compose exec postgres pg_dump -U relay relay > relay-backup.sql
docker run --rm -v relay_relaydata:/data -v "$PWD":/backup alpine tar czf /backup/relaydata.tgz -C /data .
```

The `relaydata` volume holds the key pepper. **If you restore the database but lose the pepper, every API key stops working.** Relay notices this: it refuses to start when it would generate a new pepper next to keys that already exist, and tells you why.

## More than one gateway

Gateways are stateless; durable state is in Postgres and Redis. To run several behind a load balancer:

- give every one the same `RELAY_DATABASE_URL`, `RELAY_REDIS_URL` and `RELAY_KEY_PEPPER` (the pepper must be set explicitly, since a generated one lives in one volume);
- set `RELAY_REPLICA_COUNT` to how many there are, so per-replica limits (concurrency) are divided correctly;
- keep Redis reachable from all of them. Relay refuses to start with `RELAY_REPLICA_COUNT` above 1 and no Redis, because rate limits and budgets would then be wrong.

Circuit breakers and request queues are per gateway by design; budgets, rate limits and provider pauses are shared through Redis.

## Check that it works

```sh
make smoke        # brings the stack up in an isolated project, runs end-to-end checks, tears it down
```

It checks that the first key is printed, that a request without a key is refused, that one with the key is answered, that the request was recorded in Postgres, and that metrics are not on the public port. CI runs it, and runs the quickstart in the README exactly as written.

## Stop and remove

```sh
docker compose down          # stop; keeps your data
docker compose down -v       # stop and delete the volumes, including the key pepper
```
