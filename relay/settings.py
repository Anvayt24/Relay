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
