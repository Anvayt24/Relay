"""Process-level settings, read from environment variables (prefix ``RELAY_``) or ``.env``.

This holds only *deployment* settings: how the process runs and where its dependencies
are. Business configuration (models, prices, routes, tenants) is a separate, validated
YAML config. Keeping the two apart means changing a price never
requires a redeploy-style env change, and secrets never end up in the YAML.
"""

from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from relay.auth.secrets import load_or_create_secret


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RELAY_",
        env_file=".env",
        # .env also holds provider keys (GROQ_API_KEY, ...) that are read elsewhere.
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    # json: one JSON object per line with request_id and trace_id (compose); text: readable.
    log_format: Literal["json", "text"] = "text"
    # OpenTelemetry. Unset: tracing is off (OTel's no-op tracer, zero cost).
    # Example: http://otel-collector:4317 (OTLP gRPC).
    otel_endpoint: str | None = None
    # Share of new traces recorded (a caller's traceparent decision is respected).
    otel_sample_ratio: float = 1.0
    # The adapters' read timeout for a non-streaming call. The tenant's deadline
    # (config `deadlines`, enforced by relay/reliability) usually ends a request sooner.
    request_timeout_s: float = 60.0
    # Streaming. First token: an operator ceiling on how long any one
    # attempt waits for its first content chunk; the tenant's `deadlines.first_token_s` and
    # the time left can only shorten it. Idle: the longest allowed silence *between* chunks
    # once streaming has started.
    first_token_timeout_s: float = 15.0
    stream_idle_timeout_s: float = 30.0
    # TCP + TLS handshake with a provider. Short on purpose: a provider that can't even
    # accept a connection in 5 s is down, and falling back beats waiting.
    connect_timeout_s: float = 5.0

    # Serve /metrics on this port instead of the API's. Metrics name tenants, models
    # and spend, so a deployment keeps this port off the network and lets Prometheus scrape it
    # internally. Unset: the API serves /metrics itself, as before.
    metrics_port: int | None = Field(default=None, gt=0, lt=65536)
    # The demo page and its backend. The page is served at `/` when `ui` is on. The
    # keyless chat endpoints exist only for the tenants named here: `demo_tenant` answers with
    # the real (free) models, `demo_sim_tenant` with the built-in simulator, so the router and
    # the cache are visible without any provider key.
    ui: Literal["on", "off"] = "on"
    # The built page (index.html, style.css, dist/). Unset: `web/` beside the working directory.
    web_dir: str | None = None
    demo_tenant: str | None = None
    demo_sim_tenant: str | None = None
    # The chaos panel breaks the in-process simulator (needs `sim_scenario`) and compares
    # Relay with a naive client. Relay's own side of the comparison runs as this tenant.
    demo_chaos: Literal["on", "off"] = "off"
    demo_chaos_tenant: str = "demo-chaos"
    # The largest request body accepted, in bytes (413 above it). Checked before the body is
    # read, so a huge upload costs nothing. 1 MiB is far above any chat request.
    max_body_bytes: int = Field(default=1_048_576, gt=0)
    # Addresses or CIDR blocks of the reverse proxies in front of Relay, comma-separated. Only
    # their X-Forwarded-For is believed (per-IP limits); empty trusts no header.
    trusted_proxies: str = ""

    # Request logs. With no database_url, rows go to a bounded in-memory sink.
    # Example: postgresql://relay:relay@localhost:5432/relay
    database_url: str | None = None
    log_queue_max: int = 10_000
    log_batch_max: int = 500
    log_flush_interval_s: float = 0.5
    # How long shutdown may spend flushing queued rows before giving up on them.
    log_shutdown_timeout_s: float = 5.0
    # How long shutdown waits for open streams to finish (after uvicorn's own graceful
    # shutdown) before closing the stores they settle against.
    shutdown_drain_s: float = 10.0

    # Prompt embeddings for the semantic cache (and the router). fastembed loads
    # BAAI/bge-small-en-v1.5 (ONNX) in the background; "hash" is a cheap fake for tests and
    # quick local runs; "off" disables the semantic layer. Only loaded when some tenant has
    # `cache.semantic` on.
    embedder: Literal["fastembed", "hash", "off"] = "fastembed"
    # Threads ONNX Runtime uses inside one embedding. Its default (every visible core)
    # oversubscribes a CPU-limited container and is ~10x slower: see relay/cache/embed.py.
    embedder_onnx_threads: int = 1

    # The mock provider. Unset: the scripted echo mock (deterministic, instant). Set to a
    # preset name in relay/simulator/scenario.py (e.g. "realistic"): seeded, sampled latency,
    # errors and capacity from the provider simulator.
    sim_scenario: str | None = None
    sim_seed: int = 0

    # The YAML config (providers, models, prices, routes, tenants). None = the shipped
    # relay/config/relay.yaml.
    config_path: str | None = None

    # API-key auth. The pepper is the HMAC secret for key hashes: required whenever
    # auth is on, and must be the same on every replica and in the key CLI.
    key_pepper: SecretStr | None = None
    # Skip auth and treat every request as `auth_disabled_tenant`. For local runs and tests
    # only: refused at startup unless env is dev or test.
    auth_disabled: bool = False
    auth_disabled_tenant: str = "dev"

    # Admission state shared across replicas: rate limits, budgets, provider pauses.
    # Unset: in-process state, correct for exactly one replica.
    # Example: redis://localhost:6379/0
    redis_url: str | None = None
    # How many gateway replicas share the tenant limits. Per-replica state (concurrency
    # slots, and rate limits while Redis is down) gets limit / replica_count each, so the
    # fleet as a whole stays within the configured limits.
    replica_count: int = 1
    # The priority queue: how long each class may wait for a slot, and how
    # many may wait at once.
    queue_max_wait_interactive_s: float = 2.0
    queue_max_wait_batch_s: float = 30.0
    queue_max_depth: int = 100
    # How often each replica re-reads the route cards the evidence worker writes.
    # A card change (an edge turning unsafe) reaches every replica within this interval.
    card_poll_s: float = 5.0
    # A card the worker hasn't re-stamped for this long is ignored (the worker is down, so its
    # state may be out of date): its edge is treated as undecided. 20 worker runs at 30 s.
    card_max_age_s: float = 600.0
    # How often each replica re-reads the policy pointer: a rollback reaches every
    # replica within this interval.
    policy_poll_s: float = 5.0
    # Bearer token for /v1/admin/policy. Unset: the admin API answers 404 (off unless asked).
    admin_token: SecretStr | None = None
    # A directory that survives restarts (a volume, `/data` in Docker). Secrets not set in the
    # environment are generated here once and reused: the key pepper always, the
    # admin token when `admin_api` is on. Unset: nothing is generated, as before.
    data_dir: str | None = None
    # "on" turns the admin API on, creating its token in `data_dir` if none is set.
    admin_api: Literal["on", "off"] = "off"
    # True when this process just generated the pepper (the first start on this volume). The
    # app checks then that no API key already exists: keys made with a lost pepper would all
    # stop matching, silently.
    key_pepper_generated: bool = False
    # On the first start with auth on and no API key yet, create one for this tenant and print
    # it once. Needs a database.
    bootstrap_key: bool = True
    bootstrap_tenant: str = "dev"

    @model_validator(mode="after")
    def _generate_missing_secrets(self) -> Self:
        if self.data_dir is None:
            return self
        if self.key_pepper is None:
            value, created = load_or_create_secret(self.data_dir, "key_pepper")
            self.key_pepper = SecretStr(value)
            self.key_pepper_generated = created
        if self.admin_api == "on" and self.admin_token is None:
            self.admin_token = SecretStr(load_or_create_secret(self.data_dir, "admin_token")[0])
        return self
