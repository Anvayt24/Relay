"""Which adapters exist at startup: exactly the providers whose key is set. No network here:
building an adapter only builds an httpx client, which connects lazily."""

from pydantic import SecretStr

from relay.config.loader import load_config
from relay.config.models import available_models
from relay.providers.anthropic import AnthropicAdapter
from relay.providers.openai_compat import OpenAICompatAdapter
from relay.providers.registry import NO_KEYS, ProviderKeys, build_adapters
from relay.settings import Settings

CONFIG = load_config()  # the shipped relay.yaml: all five providers


async def test_only_providers_with_keys_get_adapters() -> None:
    keys = ProviderKeys(
        _env_file=None,  # type: ignore[call-arg]
        groq_api_key=SecretStr("g"),
        anthropic_api_key=SecretStr("a"),
    )
    adapters = build_adapters(keys, Settings(_env_file=None), CONFIG.providers)  # type: ignore[call-arg]
    assert set(adapters) == {"groq", "anthropic"}
    assert isinstance(adapters["groq"], OpenAICompatAdapter)
    assert isinstance(adapters["anthropic"], AnthropicAdapter)
    for adapter in adapters.values():
        await adapter.aclose()


def test_no_keys_means_only_mock_models_are_available() -> None:
    adapters = build_adapters(NO_KEYS, Settings(_env_file=None), CONFIG.providers)  # type: ignore[call-arg]
    assert adapters == {}
    available = available_models(CONFIG.models, {"mock": None, **adapters})
    assert set(available) == {"mock/cheap", "mock/strong"}


def test_empty_key_counts_as_missing() -> None:
    keys = ProviderKeys(_env_file=None, groq_api_key=SecretStr(""))  # type: ignore[call-arg]
    assert keys.get("GROQ_API_KEY") is None


def test_keys_never_show_in_repr() -> None:
    keys = ProviderKeys(_env_file=None, groq_api_key=SecretStr("super-secret"))  # type: ignore[call-arg]
    assert "super-secret" not in repr(keys)
