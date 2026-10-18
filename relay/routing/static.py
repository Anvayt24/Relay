"""Router v0: static rules (PRD FR-7). An alias maps to a fixed model; an explicit model name
is used as-is.

Lookups use the full Relay name as the key and never split it. Upstream IDs contain slashes
(``openrouter/meta-llama/<model>:free``), so any code that splits a name has to guess where
the provider prefix ends. Looking up the whole name removes the guess.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from relay.config.models import ModelSpec
from relay.routing.base import RouteDecision
from relay.schemas.chat import ChatRequest
from relay.schemas.errors import ErrorClass, RelayError

if TYPE_CHECKING:
    from relay.pipeline.context import RequestContext

STATIC_POLICY_VERSION = "static-v0"


class StaticRouter:
    def __init__(self, models: Mapping[str, ModelSpec], aliases: Mapping[str, str]) -> None:
        # A dangling alias is a config bug. Catch it when the app starts, not when the first
        # unlucky client happens to use it.
        for alias, target in aliases.items():
            if target not in models:
                raise ValueError(f"alias {alias!r} points at unknown model {target!r}")
        self._models = models
        self._aliases = aliases

    async def decide(self, req: ChatRequest, ctx: "RequestContext") -> RouteDecision:
        if req.model in self._aliases:
            name, reason = self._aliases[req.model], "alias"
        elif req.model in self._models:
            name, reason = req.model, "explicit"
        else:
            raise RelayError(
                404,
                ErrorClass.MODEL_NOT_FOUND,
                f"model {req.model!r} does not exist; GET /v1/models lists the available ones",
                param="model",
            )
        return RouteDecision(
            tier=self._models[name].tier,
            model=name,
            score=None,
            reason=reason,
            policy_version=STATIC_POLICY_VERSION,
        )
