"""What the cache decided for one request, carried on the context for headers and logs.

Kept in its own module because ``RequestContext`` imports it and the cache stage imports the
context: a shared leaf module avoids the cycle.
"""

from dataclasses import dataclass
from typing import Literal

Status = Literal["hit", "miss", "bypass"]
Layer = Literal["exact", "semantic"]


@dataclass
class CacheResult:
    # hit: answered from the cache. miss: looked, found nothing (the answer may be stored).
    # bypass: not looked at all (ineligible, opted out, or the cache was unavailable).
    status: Status
    layer: Layer | None = None
    served_model: str | None = None  # on a hit: the model that originally produced the answer
    similarity: float | None = None  # on a semantic hit
    # On a miss, set when the finished answer should be written back.
    exact_key: str | None = None
    namespace: str | None = None
    guard: str | None = None
    ttl_s: int = 0

    @property
    def hit(self) -> bool:
        return self.status == "hit"
