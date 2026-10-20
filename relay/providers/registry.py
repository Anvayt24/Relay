"""Builds the provider adapters at startup, one per provider whose API key is set.

Keys come from unprefixed env vars (``GROQ_API_KEY``, ...), the names providers' own docs and
tools use, so a developer's existing ``.env`` just works. They are held as ``SecretStr``, so
an accidental ``print(keys)`` or a traceback shows ``**********``, not the key.

A provider without a key is simply not registered, and its models drop out of the registry
(``relay.config.models.available_models``). Startup never fails over a missing key: a developer
with only a Groq key still gets a working gateway.
"""

import logging
from collections.abc import Mapping

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from relay.config.models import ProviderSpec
from relay.providers.anthropic import AnthropicAdapter
from relay.providers.base import ProviderAdapter
from relay.providers.openai_compat import OpenAICompatAdapter
from relay.settings import Settings

logger = logging.getLogger(__name__)


class ProviderKeys(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    def get(self, env_name: str) -> str | None:
        value = getattr(self, env_name.lower(), None)
        if isinstance(value, SecretStr) and value.get_secret_value():
            return value.get_secret_value()
        return None


NO_KEYS = ProviderKeys(_env_file=None)


def build_adapters(
    keys: ProviderKeys, settings: Settings, providers: Mapping[str, ProviderSpec]
) -> dict[str, ProviderAdapter]:
    adapters: dict[str, ProviderAdapter] = {}
    skipped = []
    for name, spec in providers.items():
        if spec.kind == "mock":
            continue  # the mock adapter is built by the app factory (it may be simulated)
        key = keys.get(spec.api_key_env)
        if key is None:
            skipped.append(name)
            continue
        adapters[name] = _adapter(name, spec, key, settings)
    if skipped:
        logger.info("no API key set for provider(s) %s; their models are unavailable", skipped)
    return adapters


def _adapter(name: str, spec: ProviderSpec, key: str, settings: Settings) -> ProviderAdapter:
    if spec.kind == "anthropic":
        return AnthropicAdapter(
            name,
            base_url=spec.base_url,
            api_key=key,
            limit_scope=spec.limit_scope,
            connect_timeout_s=settings.connect_timeout_s,
            read_timeout_s=settings.request_timeout_s,
            stream_idle_timeout_s=settings.stream_idle_timeout_s,
        )
    return OpenAICompatAdapter(
        name,
        base_url=spec.base_url,
        api_key=key,
        limit_scope=spec.limit_scope,
        max_tokens_param=spec.max_tokens_param,
        connect_timeout_s=settings.connect_timeout_s,
        read_timeout_s=settings.request_timeout_s,
        stream_idle_timeout_s=settings.stream_idle_timeout_s,
    )
