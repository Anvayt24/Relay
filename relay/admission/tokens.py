"""Token estimates for reservations (PRD FR-4; ADR 0006 answers PRD §15 Q13).

Before a request runs, Relay must reserve TPM and money for it, so it needs a token count it
can't know yet. The estimate:

    prompt ≈ ceil(UTF-8 bytes / 3) + 4 per message, then x 1.10 (safety margin)
    output = max_tokens if the client set it, else the model's default_max_tokens

Why bytes / 3, not a real tokenizer:
- providers use *different* tokenizers (OpenAI's o200k, Llama's, Anthropic's own), so no
  single tokenizer is right for all of them anyway;
- English averages about 4 bytes per token, so bytes / 3 *over*-estimates it by about 30%. That
  is the safe direction: a reservation that is too big is released at reconcile, while one that
  is too small lets spend overshoot the budget;
- code and non-Latin scripts have more tokens per byte, but they also have more bytes per
  character (CJK is 3 bytes each in UTF-8), which keeps bytes / 3 roughly safe;
- no dependency, no vocabulary file to download, and nanoseconds per request.

Misses are counted (``relay_reservation_underestimate_total``) so the margin can be tuned on
evidence, not guessed.
"""

import math

from relay.config.models import ModelSpec
from relay.schemas.chat import ChatRequest

BYTES_PER_TOKEN = 3
PER_MESSAGE_OVERHEAD = 4  # role markers and separators that every chat template adds
SAFETY_MARGIN = 1.10


def text_tokens(text: str) -> int:
    return math.ceil(len(text.encode("utf-8")) / BYTES_PER_TOKEN)


def prompt_tokens(req: ChatRequest) -> int:
    raw = sum(text_tokens(m.content) + PER_MESSAGE_OVERHEAD for m in req.messages)
    return math.ceil(raw * SAFETY_MARGIN)


def output_cap(req: ChatRequest, model: ModelSpec) -> int:
    return req.max_tokens or model.default_max_tokens
