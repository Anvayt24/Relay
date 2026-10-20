"""The model registry: which Relay model names exist and where they are served.

In code for now; Phase 5 loads it from YAML with validation (PRD §9.5). The shape already
matches the YAML: a Relay name maps to (provider, provider_model, tier).

Relay names and upstream IDs are kept apart on purpose:
- clients only ever see Relay names;
- adapters only ever see ``provider_model``.

That lets an upstream rename (common with free tiers) be a one-line config change instead of a
breaking change for every client.
"""

from dataclasses import dataclass
from typing import Literal

Tier = Literal["cheap", "strong"]


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    provider_model: str
    tier: Tier


MODELS: dict[str, ModelSpec] = {
    "mock/cheap": ModelSpec(provider="mock", provider_model="mock-cheap-v1", tier="cheap"),
    "mock/strong": ModelSpec(provider="mock", provider_model="mock-strong-v1", tier="strong"),
}

# Alias -> Relay model name. Per-tenant routes for aliases arrive in Phase 5 (FR-3).
ALIASES: dict[str, str] = {
    "relay/cheap": "mock/cheap",
    "relay/strong": "mock/strong",
    # No trained router exists yet. FR-7 says a missing router artifact routes to strong:
    # when unsure, spend money rather than quality.
    "relay/auto": "mock/strong",
}
