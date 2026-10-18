"""Process-level settings, read from environment variables (prefix ``RELAY_``) or ``.env``.

This holds only *deployment* settings: how the process runs and where its dependencies
are. Business configuration (models, prices, routes, tenants) is a separate, validated
YAML config that arrives in Phase 5. Keeping the two apart means changing a price never
requires a redeploy-style env change, and secrets never end up in the YAML.
"""

from typing import Literal

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

    # Request logs. With no database_url, rows go to a bounded in-memory sink.
    # Example: postgresql://relay:relay@localhost:5432/relay
    database_url: str | None = None
    log_queue_max: int = 10_000
    log_batch_max: int = 500
    log_flush_interval_s: float = 0.5
    # How long shutdown may spend flushing queued rows before giving up on them.
    log_shutdown_timeout_s: float = 5.0
