"""What the router looks at (PRD FR-7): the meaning of the prompt, plus a few cheap facts.

A feature vector is two blocks laid end to end:

1. the **embedding** of the last user message (384 numbers; ``relay/cache/embed.py``), which
   carries "what kind of request is this?": trivia, arithmetic, code, open-ended advice;
2. a handful of **scalar features** the embedding doesn't capture well: how long the prompt is,
   how many turns the conversation has, whether it contains a code fence, whether the request
   uses tools, and which writing system it is in.

Training (``evals/router/``) and serving (``TrainedRouter``) must compute *exactly* the same
vector, or the weights mean nothing. So both call the functions here, and the artifact records
the feature spec version: a router trained against a different layout is refused at load time.

The text is normalized the same way the cache normalizes it (``keys.normalize``), so when the
cache stage has already embedded a single-user-message prompt, that embedding is the one this
module would compute, and can be reused.
"""

import math

import numpy as np

from relay.cache.keys import normalize
from relay.evidence.cohorts import script_of
from relay.schemas.chat import ChatRequest

FEATURE_SPEC_VERSION = 1

SCRIPTS = ("latin", "devanagari", "cjk", "cyrillic", "arabic", "other")
SCALAR_NAMES = (
    "log_chars",
    "turns",
    "has_code_fence",
    "has_tools",
    *(f"script_{s}" for s in SCRIPTS),
)
N_SCALARS = len(SCALAR_NAMES)


def last_user_text(req: ChatRequest) -> str:
    """The last user message, normalized. FR-7: the router decides from what is being asked
    *now*, not from earlier turns. A request with no user message gets the empty string."""
    for message in reversed(req.messages):
        if message.role == "user":
            return normalize(message.content)
    return ""


def count_turns(req: ChatRequest) -> int:
    return sum(1 for m in req.messages if m.role == "user")


def scalar_features(text: str, *, turns: int = 1, tools: bool = False) -> list[float]:
    """The raw (not yet standardized) scalar block, in ``SCALAR_NAMES`` order."""
    script = script_of(text)
    script = script if script in SCRIPTS else "other"
    return [
        math.log1p(len(text)),
        float(turns),
        float("```" in text),
        float(tools),
        *(float(script == s) for s in SCRIPTS),
    ]


def request_scalars(req: ChatRequest) -> tuple[str, list[float]]:
    """(the text the embedding is computed from, its scalar features) for a request."""
    text = last_user_text(req)
    # Tools are rejected at P0 (FR-1), so this is always False today; the slot exists so the
    # layout won't change when tool pass-through arrives.
    return text, scalar_features(text, turns=count_turns(req), tools=False)


def standardize(scalars: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per scalar (``scale`` has no zeros: the trainer guarantees it).
    Without this, ``log_chars`` (about 5) would dominate the 0/1 flags and the L2 penalty would
    treat the features unequally."""
    standardized: np.ndarray = (scalars - mean) / scale
    return standardized


def feature_vector(
    embedding: np.ndarray, scalars: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return np.concatenate([embedding, standardize(scalars, mean, scale)])
