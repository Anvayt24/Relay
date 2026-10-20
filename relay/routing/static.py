"""Router v0: static rules (PRD FR-7). An alias maps to a fixed model through the tenant's
route; an explicit model name is used as-is.

Lookups use the full Relay name as the key and never split it. Upstream IDs contain slashes
(``openrouter/meta-llama/<model>:free``), so any code that splits a name has to guess where
the provider prefix ends. Looking up the whole name removes the guess.

Aliases are per route (FR-3): ``relay/cheap`` is a simulator model on the ``default`` route
and a Groq model on ``live``. The tenant decides the route, so two tenants sending the same
request can reach different models.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from relay.config.models import ModelSpec
from relay.config.schema import RelayConfig
from relay.routing.base import RouteDecision
from relay.schemas.chat import ChatRequest
from relay.schemas.errors import ErrorClass, RelayError

if TYPE_CHECKING:
    from relay.pipeline.context import RequestContext

STATIC_POLICY_VERSION = "static-v0"


def model_not_found(name: str) -> RelayError:
    # One message for "doesn't exist", "not on this tenant's list" and "provider not
    # configured", so a caller can't map out the catalogue by probing names.
    return RelayError(
        404,
        ErrorClass.MODEL_NOT_FOUND,
        f"model {name!r} does not exist or you do not have access to it; "
        "GET /v1/models lists the ones you can use",
        param="model",
    )


class StaticRouter:
    def __init__(self, config: RelayConfig, available: Mapping[str, ModelSpec]) -> None:
        # Aliases were validated when the config loaded; nothing to check here.
        self._config = config
        self._available = available  # models whose provider has an adapter (a key)

    async def decide(self, req: ChatRequest, ctx: "RequestContext") -> RouteDecision:
        aliases = self._config.routes[ctx.route].aliases
        if req.model in aliases:
            name, reason = aliases[req.model], "alias"
        elif req.model in self._config.models:
            name, reason = req.model, "explicit"
        else:
            raise model_not_found(req.model)
        if name not in self._available:
            raise model_not_found(req.model)
        return RouteDecision(
            tier=self._config.models[name].tier,
            model=name,
            score=None,
            reason=reason,
            policy_version=STATIC_POLICY_VERSION,
        )
