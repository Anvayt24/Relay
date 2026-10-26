"""Which requests may use the cache at all (PRD FR-6 "Eligibility").

Caching an answer is only correct when the same request would get the same answer, so the
rules are conservative:

- ``temperature == 0``. Sampling at temperature > 0 is *meant* to vary. An *unset* temperature
  is not eligible unless the tenant opts in, because OpenAI's default is 1, not 0.
- no tools (rejected at P0 anyway, kept explicit), ``n == 1``;
- the prompt is not too large (a huge prompt is a poor cache candidate and a big stored value);
- the client didn't say ``X-Relay-Cache: no-store``.

The semantic layer is stricter still: exactly one user message (plus an optional system
prompt), because matching a multi-turn conversation by the similarity of its last message
would ignore everything before it.
"""

from dataclasses import dataclass
from typing import Literal

from relay.admission.tokens import prompt_tokens
from relay.cache.keys import user_text
from relay.config.tenants import TenantCache
from relay.schemas.chat import ChatRequest

Reason = Literal[
    "ok",
    "disabled",
    "no_store",
    "temperature",
    "n",
    "too_large",
]


@dataclass(frozen=True)
class Eligibility:
    exact: bool
    semantic: bool
    reason: Reason  # why not, when exact is False


def check(req: ChatRequest, policy: TenantCache, *, no_store: bool) -> Eligibility:
    if not policy.enabled:
        return Eligibility(False, False, "disabled")
    if no_store:
        return Eligibility(False, False, "no_store")
    if req.temperature is None:
        if not policy.allow_unset_temperature:
            return Eligibility(False, False, "temperature")
    elif req.temperature != 0:
        return Eligibility(False, False, "temperature")
    if req.n not in (None, 1):
        return Eligibility(False, False, "n")
    if prompt_tokens(req) > policy.max_prompt_tokens:
        return Eligibility(False, False, "too_large")
    return Eligibility(True, policy.semantic and user_text(req) is not None, "ok")
