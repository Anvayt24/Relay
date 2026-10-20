"""Relay with the simulator as its in-process mock provider (RELAY_SIM_SCENARIO)."""

from pathlib import Path

import httpx2
import openai
import pytest

from relay.api.app import create_app
from relay.providers.registry import NO_KEYS
from relay.schemas.chat import ChatRequest
from relay.settings import Settings
from relay.simulator.engine import Simulator
from relay.simulator.scenario import PRESETS


def _settings(**overrides: object) -> Settings:
    # Like conftest's test_settings: never read the developer's .env; auth off, test tenant.
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        env="test",
        database_url=None,
        config_path=str(Path(__file__).parents[1] / "config" / "test.yaml"),
        auth_disabled=True,
        auth_disabled_tenant="test",
        **overrides,  # type: ignore[arg-type]
    )


async def test_relay_streams_the_seeded_simulator_reply() -> None:
    app = create_app(_settings(sim_scenario="realistic", sim_seed=0), provider_keys=NO_KEYS)
    http_client = openai.DefaultAsyncHttpxClient(transport=httpx2.ASGITransport(app=app))
    async with openai.AsyncOpenAI(
        base_url="http://relay.test/v1", api_key="test-key", http_client=http_client
    ) as sdk:
        # max_tokens keeps the realistic reply (and so the test) short.
        stream = await sdk.chat.completions.create(
            model="relay/strong",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=3,
            stream=True,
        )
        text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])

    # Same preset, same seed, request index 0: the reply Relay streamed is exactly the one
    # the simulator's plan says, so a run is reproducible end to end.
    expected = Simulator.from_scenario(PRESETS["realistic"], seed=0).open(
        ChatRequest.model_validate(
            {
                "model": "mock-strong-v1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
            }
        ),
        index=0,
    )
    assert expected.plan.error is None
    assert text == " ".join(expected.plan.reply_words)


def test_unknown_scenario_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="RELAY_SIM_SCENARIO"):
        create_app(_settings(sim_scenario="nope"), provider_keys=NO_KEYS)
