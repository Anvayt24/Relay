"""Does the simulator's OpenAI wire format match the real OpenAI API? (PRD §13 risk:
"simulator results don't reflect reality".)

This compares **shape, not content**: field names, chunk order, error envelope and header
names. Every field the simulator sends must also exist in the real response. The simulator
may omit fields (real responses carry extras such as ``system_fingerprint``); it must not
invent any, because an adapter written against an invented field would break in production.

**It calls the real API.** It is marked ``live``, so ``make check`` and CI never run it.
Run it by hand, only with the user's OK. It makes exactly three tiny calls:
1. a non-streaming completion capped at 5 output tokens;
2. the same, streamed;
3. an unknown model, which fails with 404 before generating anything.

    $env:RELAY_LIVE_OPENAI_MODEL = "<cheapest model>"   # PowerShell
    uv run pytest -m live tests/simulator/test_wire_fidelity_live.py

The key is read from ``OPENAI_API_KEY`` (environment or ``.env``) and never printed.
"""

import json
import os
from typing import Any

import httpx2
import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

from relay.simulator.engine import Simulator
from relay.simulator.scenario import Capacity, ModelProfile, OutputLength, Scenario
from relay.simulator.server import create_sim_app

pytestmark = pytest.mark.live

PROMPT = [{"role": "user", "content": "Say hi."}]
SIM_MODEL = "sim-model"


class _LiveSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: str | None = None


Exchange = tuple[int, dict[str, str], Any]


async def _post(client: httpx2.AsyncClient, body: dict[str, Any]) -> Exchange:
    """POST a chat completion; return (status, headers, JSON body or list of SSE payloads)."""
    response = await client.post("/v1/chat/completions", json=body)
    headers = {k.lower(): v for k, v in response.headers.items()}
    if body.get("stream") and response.status_code == 200:
        events = [
            line.removeprefix("data: ")
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        return response.status_code, headers, events
    return response.status_code, headers, response.json()


async def _three_exchanges(client: httpx2.AsyncClient, model: str) -> list[Exchange]:
    base = {"model": model, "messages": PROMPT, "max_completion_tokens": 5}
    return [
        await _post(client, base),
        await _post(client, {**base, "stream": True, "stream_options": {"include_usage": True}}),
        await _post(client, {**base, "model": "relay-fidelity-check-no-such-model"}),
    ]


def _kind(chunk: dict[str, Any]) -> str:
    if not chunk["choices"]:
        return "usage"
    choice = chunk["choices"][0]
    if choice.get("finish_reason"):
        return "finish"
    return "role" if "role" in choice["delta"] else "content"


def _collapse(kinds: list[str]) -> list[str]:
    return [k for i, k in enumerate(kinds) if i == 0 or kinds[i - 1] != k]


def _assert_subset(sim: dict[str, Any], real: dict[str, Any], where: str) -> None:
    extra = set(sim) - set(real)
    assert not extra, f"{where}: the simulator invents {sorted(extra)}"


async def test_simulator_wire_shape_matches_openai() -> None:
    model = os.environ.get("RELAY_LIVE_OPENAI_MODEL")
    key = _LiveSettings().openai_api_key
    if not model or not key:
        pytest.skip("set RELAY_LIVE_OPENAI_MODEL and OPENAI_API_KEY to run")

    async with httpx2.AsyncClient(
        base_url="https://api.openai.com",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    ) as client:
        real = await _three_exchanges(client, model)

    scenario = Scenario(
        models={SIM_MODEL: ModelProfile(length=OutputLength(4, 8))},
        capacity=Capacity(rpm=1000, tpm=1_000_000),
    )
    app = create_sim_app(Simulator.from_scenario(scenario))
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://sim.test"
    ) as client:
        sim = await _three_exchanges(client, SIM_MODEL)

    # 1. Non-streaming body, field by field.
    (real_status, real_headers, real_body), (sim_status, sim_headers, sim_body) = real[0], sim[0]
    assert (real_status, sim_status) == (200, 200)
    _assert_subset(sim_body, real_body, "completion")
    _assert_subset(sim_body["choices"][0], real_body["choices"][0], "choice")
    _assert_subset(sim_body["choices"][0]["message"], real_body["choices"][0]["message"], "message")
    _assert_subset(sim_body["usage"], real_body["usage"], "usage")
    sim_limits = {h for h in sim_headers if h.startswith("x-ratelimit-")}
    assert sim_limits <= set(real_headers), sorted(sim_limits - set(real_headers))

    # 2. Stream: same order of chunk kinds, and no invented fields in any kind of chunk.
    (_, _, real_events), (_, _, sim_events) = real[1], sim[1]
    assert real_events[-1] == sim_events[-1] == "[DONE]"
    real_chunks = [json.loads(e) for e in real_events[:-1]]
    sim_chunks = [json.loads(e) for e in sim_events[:-1]]
    assert _collapse([_kind(c) for c in sim_chunks]) == _collapse([_kind(c) for c in real_chunks])
    for kind in ("role", "content", "finish", "usage"):
        real_keys = set().union(*(c.keys() for c in real_chunks if _kind(c) == kind))
        for chunk in (c for c in sim_chunks if _kind(c) == kind):
            _assert_subset(chunk, dict.fromkeys(real_keys), f"{kind} chunk")

    # 3. Error envelope for an unknown model.
    (real_status, _, real_error), (sim_status, _, sim_error) = real[2], sim[2]
    assert real_status == sim_status == 404
    assert set(sim_error["error"]) == set(real_error["error"])
    assert sim_error["error"]["type"] == real_error["error"]["type"]
    assert sim_error["error"]["code"] == real_error["error"]["code"]
