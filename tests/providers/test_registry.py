"""Which adapters exist at startup: exactly the providers whose key is set. No network here:
building an adapter only builds an httpx client, which connects lazily."""

from pydantic import SecretStr

from relay.config.loader import load_config
from relay.config.models import available_models
from relay.providers.anthropic import AnthropicAdapter
from relay.providers.openai_compat import OpenAICompatAdapter
from relay.providers.registry import NO_KEYS, ProviderKeys, build_adapters
from relay.schemas.chat import ChatRequest
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


def test_models_marked_thinking_off_reach_the_anthropic_adapter() -> None:
    # relay.yaml marks a model `thinking: off`; the adapter is per provider, so the registry
    # hands it the upstream ids of that provider's models that ask for it, and no others.
    keys = ProviderKeys(_env_file=None, anthropic_api_key=SecretStr("a"))  # type: ignore[call-arg]
    adapters = build_adapters(
        keys,
        Settings(_env_file=None),
        CONFIG.providers,
        models=CONFIG.models,  # type: ignore[call-arg]
    )
    anthropic = adapters["anthropic"]
    assert isinstance(anthropic, AnthropicAdapter)
    sonnet = CONFIG.models["anthropic/sonnet"]
    haiku = CONFIG.models["anthropic/haiku"]
    assert sonnet.thinking == "off" and haiku.thinking == "default"
    request = ChatRequest.model_validate(
        {"model": sonnet.provider_model, "messages": [{"role": "user", "content": "hi"}]}
    )
    assert anthropic._body(request)["thinking"] == {"type": "between_tools"}
    other = request.model_copy(update={"model": haiku.provider_model})
    assert "thinking" not in anthropic._body(other)
