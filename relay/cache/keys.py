"""Cache keys: what makes two requests "the same" (PRD FR-6).

Two keys, for the two layers:

- **Exact key:** a hash of everything that can change the answer: the tenant, the *requested*
  model alias, the normalized messages, and every output-affecting parameter at its exact
  value. Equal key, equal request.
- **Semantic namespace:** everything *except* the user's message text. Within a namespace,
  similar-enough prompts may share an answer; across namespaces, never. Tenant, alias,
  system prompt, and the output-affecting parameters all separate namespaces, so a "similar"
  prompt can't borrow an answer produced under different rules.

**Why the alias and not the routed model.** The router may pick a different model tomorrow
(a new policy), but the client asked for ``relay/auto``, and the cache is keyed by what was
asked. Each entry records the model that actually served it, so the response headers stay
honest; policy changes can flush by ``policy_version`` (Phase 13).

**Why max_tokens needs care.** A reply cut off at ``max_tokens: 50`` must never answer a
request for 1000 tokens, and a long reply must never answer a request capped at 50. The exact
key includes the exact value. The semantic namespace uses a power-of-two *bucket* of it, plus
the entry stores its completion length, and a hit is rejected if that length exceeds the
request's own cap (see ``fits``).

Hashes, not text: no prompt is stored in the cache's keys.
"""

import hashlib
import json
import re
import unicodedata
from typing import Any

from relay.schemas.chat import ChatRequest


def normalize(text: str) -> str:
    """Unicode NFC and no trailing whitespace: "café" typed two ways, or a stray trailing
    newline, are the same request. Case and inner whitespace are kept: they can matter."""
    return unicodedata.normalize("NFC", text).rstrip()


def effective_max_tokens(req: ChatRequest, default: int) -> int:
    return req.max_tokens or default


def max_tokens_bucket(max_tokens: int | None) -> str:
    """Power-of-two bucket (32, 64, 128, ...), or ``default`` when the client set none."""
    if max_tokens is None:
        return "default"
    bucket = 32
    while bucket < max_tokens:
        bucket *= 2
    return str(bucket)


def _digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _messages(req: ChatRequest) -> list[list[str]]:
    return [[m.role, normalize(m.content)] for m in req.messages]


def _stop(req: ChatRequest) -> list[str]:
    stop = req.stop
    return [] if stop is None else sorted([stop] if isinstance(stop, str) else stop)


def exact_key(tenant: str, req: ChatRequest) -> str:
    return _digest(
        {
            "t": tenant,
            "model": req.model,
            "messages": _messages(req),
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "stop": _stop(req),
            "seed": req.seed,
        }
    )


def system_prompt(req: ChatRequest) -> str:
    return "\n".join(normalize(m.content) for m in req.messages if m.role == "system")


def semantic_namespace(tenant: str, req: ChatRequest) -> str:
    return _digest(
        {
            "t": tenant,
            "model": req.model,
            "system": hashlib.sha256(system_prompt(req).encode("utf-8")).hexdigest(),
            "response_format": None,  # tools and response_format are rejected at P0 (FR-1)
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": max_tokens_bucket(req.max_tokens),
            "stop": _stop(req),
            "seed": req.seed,
        }
    )


def user_text(req: ChatRequest) -> str | None:
    """The single user message the semantic layer matches on, or None if the request isn't
    "one user message plus an optional system prompt" (multi-turn uses the exact cache only)."""
    others = [m for m in req.messages if m.role not in ("system", "user")]
    users = [m for m in req.messages if m.role == "user"]
    if others or len(users) != 1:
        return None
    return normalize(users[0].content)


def fits(completion_tokens: int, requested_cap: int) -> bool:
    """A cached reply may answer a request only if it is no longer than the request allows."""
    return completion_tokens <= requested_cap


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([A-Z][a-zA-Z]*)")


def guard_signature(prompt: str) -> str:
    """Lexical guard (FR-6, P1): a hash of the numbers and capitalized words in the prompt.

    Embeddings call "15% of 200" and "15% of 300" near-identical, and the answers differ. If
    the signatures differ the pair is forced to a miss. Sentence-initial words are dropped from
    the entity set, so "What" at the start of a question isn't an entity.
    """
    starts = {m.group(1) for m in _SENTENCE_START.finditer(prompt)}
    entities = {e for e in _ENTITY.findall(prompt) if e not in starts}
    numbers = set(_NUMBER.findall(prompt))
    return hashlib.sha256("|".join(sorted(numbers | entities)).encode("utf-8")).hexdigest()[:16]
