"""CacheStage: answer from the cache, or remember the answer for next time (PRD FR-6).

    Log → Admission → [Cache] → Route → Reservation → Provider

**Position.** After admission and before routing:
- *after admission*, so a hit still counts toward the tenant's requests-per-minute and
  concurrency (a cache can't be used to bypass rate limits), but reserves no tokens or money;
- *before the router*, so a hit never pays for routing, and so the one embedding the semantic
  layer computes can be reused by the router (Phase 11).

**On a hit** the stage short-circuits with a response (or a replayed stream), and no provider
is called, no reservation is made, and billed cost is zero. The headers say where it came from
(``X-Relay-Cache: hit``, provider ``relay-cache``, and the model that originally produced it).

**On a miss** it lets the request through and, in ``finalize``, writes the finished answer
back **in a background task**: the client never waits for a cache write. Only complete,
trustworthy answers are written: success, ``finish_reason: stop`` (not truncated), and not
served through an un-evidenced fallback edge (ADR 0010: a degraded answer should not become
the cached answer for everyone).

**Failure is a miss.** Redis or Postgres being down, a slow embedding, a model that failed to
load: each disables the affected layer and the request proceeds to the provider (degradation
matrix). Nothing in this stage can fail a request.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

from prometheus_client import Counter, Histogram

from relay.cache.eligibility import check
from relay.cache.embed import Embedder
from relay.cache.entry import CachedReply
from relay.cache.exact import CacheUnavailable, ExactStore
from relay.cache.keys import (
    exact_key,
    fits,
    guard_signature,
    semantic_namespace,
    user_text,
)
from relay.cache.result import CacheResult
from relay.cache.semantic import SemanticStore
from relay.config.schema import RelayConfig
from relay.pipeline.context import RequestContext
from relay.pipeline.stage import Continue, Outcome, ShortCircuit, StageResult
from relay.pipeline.streaming import PrimedStream
from relay.schemas.chat import (
    ChatChunk,
    ChatResponse,
    Choice,
    ChunkChoice,
    ChunkDelta,
    ResponseMessage,
)

logger = logging.getLogger(__name__)

REQUESTS = Counter(
    "relay_cache_requests_total",
    "Cache lookups by layer and result (bypass = not looked at).",
    ["layer", "result"],
)
ERRORS = Counter(
    "relay_cache_errors_total", "Cache backend failures, treated as misses.", ["layer"]
)
SIMILARITY = Histogram(
    "relay_cache_similarity",
    "Cosine similarity of semantic cache hits.",
    buckets=[0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0],
)
EMBED_SECONDS = Histogram("relay_cache_embed_seconds", "Time to embed one prompt.")
WRITES_DROPPED = Counter(
    "relay_cache_writes_dropped_total", "Background cache writes skipped (too many pending)."
)

MAX_PENDING_WRITES = 100
# Each replayed content chunk carries about this many characters, like a token or two.
_PIECE = 6


class CacheStage:
    name = "cache"

    def __init__(
        self,
        config: RelayConfig,
        *,
        exact: ExactStore | None,
        semantic: SemanticStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._config = config
        self._exact = exact
        self._semantic = semantic
        self._embedder = embedder
        self._pending: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------ lookup
    async def process(self, ctx: RequestContext) -> StageResult:
        req = ctx.request
        elig = check(req, ctx.tenant.cache, no_store=ctx.cache_no_store)
        if not elig.exact:
            ctx.cache = CacheResult("bypass")
            REQUESTS.labels(layer="none", result="bypass").inc()
            return Continue()

        config = ctx.applied.config if ctx.applied is not None else self._config
        route = config.routes[ctx.route]
        cap = req.max_tokens or max(
            (config.models[m].default_max_tokens for m in route.aliases.values()),
            default=1024,
        )
        scope = cache_scope(ctx)
        key = exact_key(scope, req)

        if self._exact is not None:
            try:
                reply = await self._exact.get(key)
            except CacheUnavailable:
                ERRORS.labels(layer="exact").inc()
            else:
                if reply is not None and fits(reply.completion_tokens, cap):
                    return self._hit(ctx, reply, "exact", key, None)
                REQUESTS.labels(layer="exact", result="miss").inc()

        namespace = guard = None
        # An embedder that is still loading (or failed to) just means no semantic layer yet.
        if (
            elig.semantic
            and self._semantic is not None
            and self._embedder is not None
            and self._embedder.ready
        ):
            text = user_text(req)
            assert text is not None  # eligibility guarantees one user message
            namespace = semantic_namespace(scope, req)
            guard = guard_signature(text)
            try:
                started = time.perf_counter()
                ctx.embedding = await self._embedder.embed(text)
                EMBED_SECONDS.observe(time.perf_counter() - started)
                hit = await self._semantic.lookup(
                    namespace,
                    ctx.embedding,
                    threshold=route.cache.semantic_threshold,
                    max_tokens=cap,
                    guard=guard if route.cache.lexical_guard else None,
                )
            except CacheUnavailable:
                ERRORS.labels(layer="semantic").inc()
            except Exception:  # an embedding failure must never fail the request
                ERRORS.labels(layer="embedding").inc()
                logger.exception("semantic cache lookup failed; serving from the provider")
            else:
                if hit is not None:
                    SIMILARITY.observe(hit.similarity)
                    return self._hit(ctx, hit.reply, "semantic", key, hit.similarity)
                REQUESTS.labels(layer="semantic", result="miss").inc()

        ctx.cache = CacheResult(
            "miss",
            exact_key=key,
            namespace=namespace,
            guard=guard,
            ttl_s=ctx.tenant.cache.ttl_s,
        )
        return Continue()

    def _hit(
        self,
        ctx: RequestContext,
        reply: CachedReply,
        layer: str,
        key: str,
        similarity: float | None,
    ) -> StageResult:
        ctx.cache = CacheResult(
            "hit",
            layer=layer,  # type: ignore[arg-type]
            served_model=reply.served_model,
            similarity=similarity,
        )
        REQUESTS.labels(layer=layer, result="hit").inc()
        ctx.usage = reply.usage()
        completion_id = f"chatcmpl-cache-{key[:16]}"
        if not ctx.stream:
            return ShortCircuit(
                ChatResponse(
                    id=completion_id,
                    created=int(time.time()),
                    model=reply.served_model,
                    choices=[
                        Choice(
                            message=ResponseMessage(content=reply.text),
                            finish_reason=reply.finish_reason,  # type: ignore[arg-type]
                        )
                    ],
                    usage=reply.usage(),
                )
            )
        return ShortCircuit(_replay(completion_id, reply))

    # ------------------------------------------------------------------ write-back
    async def finalize(self, ctx: RequestContext, outcome: Outcome) -> None:
        result = ctx.cache
        if result is None or result.status != "miss" or result.exact_key is None:
            return
        served = ctx.served
        if (
            outcome.status != "success"
            or ctx.finish_reason != "stop"
            or ctx.degraded
            or ctx.usage is None
            or served is None
            or not ctx.reply_parts
        ):
            return
        reply = CachedReply(
            text="".join(ctx.reply_parts),
            finish_reason="stop",
            prompt_tokens=ctx.usage.prompt_tokens,
            completion_tokens=ctx.usage.completion_tokens,
            cached_tokens=0,
            served_model=served.model,
            policy_version=ctx.policy_version,
            created_at=time.time(),
        )
        if len(self._pending) >= MAX_PENDING_WRITES:
            WRITES_DROPPED.inc()
            return
        task = asyncio.create_task(self._write(result, reply, ctx.embedding))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _write(
        self, result: CacheResult, reply: CachedReply, embedding: list[float] | None
    ) -> None:
        assert result.exact_key is not None
        if self._exact is not None:
            try:
                await self._exact.set(result.exact_key, reply, result.ttl_s)
            except CacheUnavailable:
                ERRORS.labels(layer="exact").inc()
        if (
            self._semantic is not None
            and embedding is not None
            and result.namespace is not None
            and result.guard is not None
        ):
            try:
                await self._semantic.insert(
                    result.namespace, embedding, reply, guard=result.guard, ttl_s=result.ttl_s
                )
            except CacheUnavailable:
                ERRORS.labels(layer="semantic").inc()

    async def drain(self, timeout_s: float = 5.0) -> None:
        """Wait for pending writes (shutdown, and tests). Never raises."""
        if self._pending:
            await asyncio.wait(list(self._pending), timeout=timeout_s)


def cache_scope(ctx: RequestContext) -> str:
    """Whose entries a request may see: its tenant's, and for a request to the route's router
    alias, only those produced under the same policy bundle (ADR 0023). A new bundle can route
    differently, so an answer chosen under the old one must not be served under it. Requests
    for an explicit model or another alias don't depend on the router and keep their entries."""
    applied = ctx.applied
    if applied is None or applied.version is None:
        return ctx.tenant.id
    router = applied.config.routes[ctx.route].router
    if router is not None and ctx.request.model == router.alias:
        return f"{ctx.tenant.id}@{applied.version}"
    return ctx.tenant.id


def _replay(completion_id: str, reply: CachedReply) -> PrimedStream:
    """A cached answer as a stream: a role chunk, the text in small pieces, a finish chunk,
    and the usage chunk. The client can't tell it from a live stream (apart from the headers)."""
    created = int(time.time())
    model = reply.served_model

    def chunk(delta: ChunkDelta | None = None, finish: str | None = None) -> ChatChunk:
        return ChatChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChunkChoice(delta=delta or ChunkDelta(), finish_reason=finish)  # type: ignore[arg-type]
            ],
        )

    pieces = [reply.text[i : i + _PIECE] for i in range(0, len(reply.text), _PIECE)] or [""]
    buffered = [
        chunk(ChunkDelta(role="assistant", content="")),
        chunk(ChunkDelta(content=pieces[0])),
    ]

    async def rest() -> AsyncGenerator[ChatChunk, None]:
        for piece in pieces[1:]:
            yield chunk(ChunkDelta(content=piece))
        yield chunk(finish=reply.finish_reason)
        yield ChatChunk(
            id=completion_id, created=created, model=model, choices=[], usage=reply.usage()
        )

    return PrimedStream(buffered=buffered, upstream=rest(), relay_model=model)
