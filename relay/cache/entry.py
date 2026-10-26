"""What the cache stores for one answer: the reply, not the prompt.

The prompt text is never stored (keys are hashes; the semantic layer stores an embedding and a
hash of the numbers/names for the lexical guard). The reply is, because it is the point.
"""

import json
from dataclasses import asdict, dataclass

from relay.schemas.chat import Usage


@dataclass(frozen=True)
class CachedReply:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    served_model: str  # the Relay model that originally produced it
    policy_version: str
    created_at: float  # unix seconds

    def usage(self) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
        )

    def dumps(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def loads(cls, blob: str | bytes) -> "CachedReply":
        return cls(**json.loads(blob))
