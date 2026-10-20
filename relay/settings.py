"""Process-level settings, read from environment variables (prefix ``RELAY_``) or ``.env``.

This holds only *deployment* settings: how the process runs and where its dependencies
are. Business configuration (models, prices, routes, tenants) is a separate, validated
YAML config that arrives in Phase 5. Keeping the two apart means changing a price never
requires a redeploy-style env change, and secrets never end up in the YAML.
"""

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RELAY_",
        env_file=".env",
        # .env also holds provider keys (GROQ_API_KEY, ...) that are read elsewhere.
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    # End-to-end budget per request (PRD FR-5 interactive default). Phase 7 enforces it.
    request_timeout_s: float = 60.0
    # Streaming (PRD FR-5 defaults). First token: how long to wait for the first content
    # chunk before giving up, while that is still an ordinary HTTP error. Idle: the longest
    # allowed silence *between* chunks once streaming has started.
    first_token_timeout_s: float = 15.0
    stream_idle_timeout_s: float = 30.0
    # TCP + TLS handshake with a provider. Short on purpose: a provider that can't even
    # accept a connection in 5 s is down, and Phase 7 would rather fall back than wait.
    connect_timeout_s: float = 5.0

    # Request logs. With no database_url, rows go to a bounded in-memory sink.
    # Example: postgresql://relay:relay@localhost:5432/relay
    database_url: str | None = None
    log_queue_max: int = 10_000
    log_batch_max: int = 500
    log_flush_interval_s: float = 0.5
    # How long shutdown may spend flushing queued rows before giving up on them.
    log_shutdown_timeout_s: float = 5.0

    # The mock provider. Unset: the scripted echo mock (deterministic, instant). Set to a
    # preset name in relay/simulator/scenario.py (e.g. "realistic"): seeded, sampled latency,
    # errors and capacity from the provider simulator.
    sim_scenario: str | None = None
    sim_seed: int = 0

    # The YAML config (providers, models, prices, routes, tenants). None = the shipped
    # relay/config/relay.yaml.
    config_path: str | None = None

    # API-key auth (FR-3). The pepper is the HMAC secret for key hashes: required whenever
    # auth is on, and must be the same on every replica and in the key CLI.
    key_pepper: SecretStr | None = None
    # Skip auth and treat every request as `auth_disabled_tenant`. For local runs and tests
    # only: refused at startup unless env is dev or test.
    auth_disabled: bool = False
    auth_disabled_tenant: str = "dev"

    # Admission state shared across replicas (Phase 6): rate limits, budgets, provider pauses.
    # Unset: in-process state, correct for exactly one replica.
    # Example: redis://localhost:6379/0
    redis_url: str | None = None
    # How many gateway replicas share the tenant limits. Per-replica state (concurrency
    # slots, and rate limits while Redis is down) gets limit / replica_count each, so the
    # fleet as a whole stays within the configured limits.
    replica_count: int = 1
    # The priority queue (FR-4 defaults): how long each class may wait for a slot, and how
    # many may wait at once.
    queue_max_wait_interactive_s: float = 2.0
    queue_max_wait_batch_s: float = 30.0
    queue_max_depth: int = 100
