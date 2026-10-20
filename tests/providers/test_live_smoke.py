"""Live smoke tests: tiny real calls to each provider whose key is set (PRD §11.3).

Marked ``live``: excluded from ``make check`` and CI. Run by hand only with the user's OK,
after confirming the cheapest current model per provider (real-api-usage rules). Each provider
gets exactly two calls, a completion and a stream, capped at 5 output tokens.

    $env:RELAY_LIVE_PROVIDERS = "groq,anthropic"     # PowerShell; which ones to call
    uv run pytest -m live tests/providers/test_live_smoke.py

Model per provider: ``RELAY_LIVE_<PROVIDER>_MODEL``, else the registry's model for it.
Keys come from the environment or ``.env`` and are never printed.
"""

import os

import pytest

from relay.config.loader import load_config
from relay.providers.registry import ProviderKeys, build_adapters
from relay.schemas.chat import ChatRequest
from relay.settings import Settings

pytestmark = pytest.mark.live

CONFIG = load_config()  # the shipped relay/config/relay.yaml
PROMPT = "Say hi."  # public and harmless: Gemini's free tier may keep prompts (PRD §13)


def _model_for(provider: str) -> str:
    override = os.environ.get(f"RELAY_LIVE_{provider.upper()}_MODEL")
    if override:
        return override
    return next(s.provider_model for s in CONFIG.models.values() if s.provider == provider)


@pytest.mark.parametrize(
    "provider", [p for p, spec in CONFIG.providers.items() if spec.kind != "mock"]
)
async def test_provider_smoke(provider: str) -> None:
    wanted = {p.strip() for p in os.environ.get("RELAY_LIVE_PROVIDERS", "").split(",") if p.strip()}
    if provider not in wanted:
        pytest.skip(f"{provider} not listed in RELAY_LIVE_PROVIDERS")
    adapters = build_adapters(
        ProviderKeys(),
        Settings(_env_file=None),
        CONFIG.providers,  # type: ignore[call-arg]
    )
    if provider not in adapters:
        pytest.skip(f"no API key for {provider}")
    adapter = adapters[provider]
    req = ChatRequest.model_validate(
        {
            "model": _model_for(provider),
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 5,
        }
    )
    try:
        response = await adapter.complete(req)
        chunks = [c async for c in adapter.stream(req)]
    finally:
        for a in adapters.values():
            await a.aclose()

    assert response.usage.completion_tokens <= 5 or provider == "gemini"
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None  # the adapter contract holds against the real provider
    assert any(c.is_finished for c in chunks)
