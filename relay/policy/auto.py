"""The one automatic policy action: roll back a canary whose card says unsafe (PRD §10.7;
ADRs 0022, 0024).

The evidence worker calls this for every card whose recommended action just became
``rollback``. It only acts if that exact version is *still* the running canary: an operator may
have promoted or ended it since the pairs were judged, and an automatic action must never undo
a human's newer decision. The event records the actor (``evidence-worker``) and the card.
Promotion is never automatic.
"""

import logging
from collections.abc import Awaitable, Callable

from relay.evidence.cards import RouteCard
from relay.policy import control
from relay.policy.control import ControlError, Pointer
from relay.policy.store import PolicyStore

logger = logging.getLogger(__name__)

ACTOR = "evidence-worker"


def rollback_unsafe_canary(store: PolicyStore) -> Callable[[RouteCard], Awaitable[None]]:
    async def hook(card: RouteCard) -> None:
        if card.recommended_action != "rollback" or not card.comparison.startswith("canary:"):
            return
        version = card.comparison.split(":", 1)[1]

        def transition(p: Pointer) -> Pointer:
            if p.candidate_version != version or p.candidate_mode != "canary":
                raise ControlError(f"canary {version} is no longer running")
            return control.rollback(p)

        reason = (
            f"canary card unsafe at look {card.look}: w={card.estimate:.4f}, ci={card.ci}, "
            f"floor {card.floor}, route {card.route}"
            if card.estimate is not None
            else f"canary card unsafe at look {card.look}"
        )
        try:
            await store.change("rollback", transition, actor=ACTOR, reason=reason)
        except ControlError as e:
            logger.info("automatic rollback skipped: %s", e)
            return
        logger.warning("canary %s rolled back automatically: %s", version, reason)

    return hook
