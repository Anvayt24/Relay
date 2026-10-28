"""FastAPI application factory.

Why a factory (``create_app()``) instead of a module-level ``app = FastAPI()``:
- Tests build a fresh app per test with their own settings and fakes. No global state
  leaks between tests.
- Importing this module has no side effects (no connections opened, no env read).
- Dependencies (providers, the log sink) are passed in as arguments, so all the wiring is
  explicit and in one place.

**Construct in the factory, start in the lifespan.** The factory only builds objects, with no
I/O and no background tasks. Anything that needs a running event loop or opens connections
(the DB pool, the log writer's task) is started in ``lifespan`` and stopped there in reverse
order.

**Shutdown drain** (PRD §8). On SIGTERM uvicorn stops accepting connections and waits for
in-flight requests (bounded by ``--timeout-graceful-shutdown``) before running lifespan
shutdown. The lifespan then marks the app as draining (``/readyz`` answers 503), waits up to
``shutdown_drain_s`` for any streams still open, and only then flushes the log writer and
closes the stores and adapters. By the time the writer flushes, every request's log row is
already queued.

Run it with ``uvicorn relay.api.app:create_app --factory``.
"""

import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from relay import __version__
from relay.admission.pause import PauseState
from relay.admission.queue import AdmissionQueue
from relay.admission.redis_store import RedisStore
from relay.admission.resilient import ResilientStore
from relay.admission.stages import AdmissionStage, ReservationStage
from relay.admission.store import CapacityStore, MemoryStore
from relay.api import health, routes
from relay.api.errors import install_error_handlers
from relay.api.request_id import RequestIdMiddleware
from relay.api.streaming import OpenStreams
from relay.auth.service import Authenticator
from relay.auth.store import CachedKeyStore, KeyStore, MemoryKeyStore, PostgresKeyStore
from relay.cache.embed import Embedder, FastEmbedder, HashEmbedder
from relay.cache.exact import ExactStore, MemoryExactStore, RedisExactStore
from relay.cache.semantic import MemorySemanticStore, PgVectorStore, SemanticStore
from relay.cache.stage import CacheStage
from relay.config.loader import DEFAULT_CONFIG, load_config
from relay.config.models import ModelSpec, available_models
from relay.config.schema import RelayConfig
from relay.config.tenants import Tenant
from relay.pipeline.runner import Pipeline
from relay.pipeline.stages import LogStage, ProviderStage, RouteStage
from relay.providers.base import ProviderAdapter
from relay.providers.mock import MockProvider
from relay.providers.registry import ProviderKeys, build_adapters
from relay.reliability.breaker import BreakerRegistry
from relay.reliability.dispatch import ReliableDispatcher
from relay.reliability.edges import SeededEdgeStates
from relay.reliability.retry_budget import RetryBudget
from relay.routing.artifact import ArtifactError, RouterArtifact
from relay.routing.artifact import load as load_artifact
from relay.routing.base import Router
from relay.routing.router import TrainedRouter
from relay.routing.static import StaticRouter
from relay.settings import Settings
from relay.simulator.engine import Simulator
from relay.simulator.scenario import PRESETS
from relay.telemetry.log_writer import LogSink, LogWriter, MemorySink
from relay.telemetry.postgres import PostgresSink

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    providers: Mapping[str, ProviderAdapter] | None = None,
    log_sink: LogSink | None = None,
    provider_keys: ProviderKeys | None = None,
    key_store: KeyStore | None = None,
    config: RelayConfig | None = None,
    capacity_store: CapacityStore | None = None,
    exact_cache: ExactStore | None = None,
    semantic_cache: SemanticStore | None = None,
    embedder: Embedder | None = None,
    router: Router | None = None,
) -> FastAPI:
    settings = settings or Settings()
    # Invalid YAML (dangling alias, cyclic fallback graph, ...) raises here: Relay won't start.
    config = config or load_config(settings.config_path)
    if providers is None:
        # Mock providers are always there (each `kind: mock` provider is its own in-process
        # simulator); real providers only when their key is set.
        keys = provider_keys if provider_keys is not None else ProviderKeys()
        providers = {
            **{
                name: _mock_provider(settings)
                for name, spec in config.providers.items()
                if spec.kind == "mock"
            },
            **build_adapters(keys, settings, config.providers),
        }
    models = available_models(config.models, providers)
    auth_disabled_tenant, authenticator = _auth(settings, config, key_store)

    postgres: PostgresSink | None = None
    memory_fallback = False
    if log_sink is None:
        if settings.database_url:
            postgres = PostgresSink(settings.database_url)
            log_sink = postgres
        else:
            log_sink = MemorySink()
            memory_fallback = True
    writer = LogWriter(
        log_sink,
        max_queue=settings.log_queue_max,
        batch_max=settings.log_batch_max,
        flush_interval_s=settings.log_flush_interval_s,
    )

    store = capacity_store or _capacity_store(settings)
    queue = AdmissionQueue(
        max_wait_s={
            "interactive": settings.queue_max_wait_interactive_s,
            "batch": settings.queue_max_wait_batch_s,
        },
        max_depth=settings.queue_max_depth,
    )

    # Built once per app, not per request. Constructors validate the config (dangling
    # aliases, models without an adapter), so a bad config fails at startup.
    # Cache backends come from settings unless a caller (a test) hands them in.
    default_exact, default_semantic = _cache_backends(settings, config)
    exact_cache = exact_cache or default_exact
    semantic_cache = semantic_cache or default_semantic
    # One embedder, shared by the semantic cache and the router (the router reuses the
    # cache's embedding when it has one). Built when either needs it.
    embedder = embedder or _embedder(settings, config)
    cache_stage = CacheStage(config, exact=exact_cache, semantic=semantic_cache, embedder=embedder)
    rel = config.reliability
    dispatcher = ReliableDispatcher(
        config,
        models,
        providers,
        store=store,
        pause=PauseState(store, config),
        breakers=BreakerRegistry(rel),
        retry_budget=RetryBudget(
            ratio=rel.retry_budget_ratio,
            minimum=rel.retry_budget_min,
            window_s=rel.retry_budget_window_s,
        ),
        edge_states=SeededEdgeStates(config),
        first_token_ceiling_s=settings.first_token_timeout_s,
    )
    pipeline = Pipeline(
        [
            LogStage(writer),  # first in, so its finalize is last out and sees the outcome
            AdmissionStage(store, queue, replica_count=settings.replica_count),
            cache_stage,  # a hit short-circuits: no routing, reservation or provider call
            RouteStage(
                router or _router(settings, config, models, embedder),
                config,
            ),
            # After routing (the price needs the model), and sized for every model the
            # dispatcher might fall back to.
            ReservationStage(
                store, config, planner=lambda ctx: [c.model for c in dispatcher.plan(ctx)]
            ),
            ProviderStage(dispatcher),
        ]
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if postgres is not None:
            await postgres.start()  # creates the pool; connects to nothing yet
        elif memory_fallback:
            logger.warning("RELAY_DATABASE_URL not set: request logs are kept in memory only")
        if authenticator is not None:
            await authenticator.store.start()
        await store.start()
        await exact_cache.start()
        if semantic_cache is not None:
            await semantic_cache.start()
        embedder_task = (
            embedder.start_in_background() if isinstance(embedder, FastEmbedder) else None
        )
        writer.start()
        try:
            yield
        finally:
            app.state.draining = True
            await _drain(app, settings.shutdown_drain_s)
            await cache_stage.drain()
            if embedder_task is not None:
                embedder_task.cancel()
            await writer.stop(settings.log_shutdown_timeout_s)
            if postgres is not None:
                await postgres.close()
            if authenticator is not None:
                await authenticator.store.close()
            await store.close()
            await exact_cache.close()
            if semantic_cache is not None:
                await semantic_cache.close()
            for adapter in providers.values():
                await adapter.aclose()  # release pooled provider connections

    app = FastAPI(title="Relay", version=__version__, lifespan=lifespan)
    # app.state is how request handlers reach process-wide objects without globals.
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.config = config
    app.state.capacity_store = store
    app.state.admission_queue = queue
    app.state.models = models
    app.state.auth_disabled_tenant = auth_disabled_tenant
    app.state.authenticator = authenticator
    app.state.log_writer = writer
    app.state.dispatcher = dispatcher
    app.state.cache_stage = cache_stage
    app.state.draining = False
    app.state.open_streams = OpenStreams()  # RelayStreamingResponse keeps this current

    install_error_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    app.include_router(routes.router)
    return app


def _cache_backends(
    settings: Settings, config: RelayConfig
) -> tuple[ExactStore, SemanticStore | None]:
    """The cache's storage, chosen from what is configured (PRD §8 degradation matrix: every
    layer is optional and fails open).

    - exact: Redis when configured, else in-process;
    - semantic: only if some tenant turned semantic caching on and the embedder isn't "off".
      Postgres/pgvector when a database is configured, else in-process.
    """
    exact: ExactStore = (
        RedisExactStore(settings.redis_url) if settings.redis_url else MemoryExactStore()
    )
    wants_semantic = any(t.cache.enabled and t.cache.semantic for t in config.tenants.values())
    if not wants_semantic or settings.embedder == "off":
        return exact, None
    semantic: SemanticStore
    if settings.database_url:
        semantic = PgVectorStore(settings.database_url)
    else:
        logger.warning("RELAY_DATABASE_URL not set: the semantic cache is in-process only")
        semantic = MemorySemanticStore()
    return exact, semantic


def _embedder(settings: Settings, config: RelayConfig) -> Embedder | None:
    """The prompt embedder, if anything needs one: a tenant with the semantic cache, or a route
    with a trained router. None when nothing does (no model is loaded for nothing), or when
    RELAY_EMBEDDER=off (then the semantic layer is skipped and the router routes to strong)."""
    needed = any(t.cache.enabled and t.cache.semantic for t in config.tenants.values()) or any(
        r.router is not None for r in config.routes.values()
    )
    if not needed or settings.embedder == "off":
        return None
    if settings.embedder == "hash":
        return HashEmbedder()
    return FastEmbedder(onnx_threads=settings.embedder_onnx_threads)


def _config_dir(settings: Settings) -> Path:
    # Router artifact paths are relative to the config file that names them.
    return Path(settings.config_path).parent if settings.config_path else DEFAULT_CONFIG.parent


def _router(
    settings: Settings,
    config: RelayConfig,
    models: Mapping[str, ModelSpec],
    embedder: Embedder | None,
) -> Router:
    """The static router, or the trained one when any route configures a ``router:``.

    A missing or invalid artifact is logged loudly and recorded as None: that route's
    relay/auto then routes to strong (PRD: "router artifact missing/corrupt -> strong") and the
    gateway still starts."""
    routed = {name: r.router for name, r in config.routes.items() if r.router is not None}
    if not routed:
        return StaticRouter(config, models)
    artifacts: dict[str, RouterArtifact | None] = {}
    for name, spec in routed.items():
        path = _config_dir(settings) / spec.artifact
        try:
            loaded = load_artifact(path)
            artifacts[name] = loaded
            logger.info("route %r: router %s from %s", name, loaded.version, path)
        except ArtifactError as e:
            artifacts[name] = None
            logger.error(
                "route %r: router artifact unusable (%s); relay/auto will route to strong", name, e
            )
    return TrainedRouter(config, models, artifacts=artifacts, embedder=embedder)


async def _drain(app: FastAPI, timeout_s: float) -> None:
    """Wait (bounded) for open streams to finish before the stores they settle against close."""
    streams: OpenStreams = app.state.open_streams
    if not await streams.wait_idle(timeout_s):
        logger.warning("shutdown with %d streams still open", streams.count)


def _mock_provider(settings: Settings) -> MockProvider:
    if settings.sim_scenario is None:
        return MockProvider()
    scenario = PRESETS.get(settings.sim_scenario)
    if scenario is None:
        # Fail at startup, not on the first request.
        raise ValueError(
            f"unknown RELAY_SIM_SCENARIO {settings.sim_scenario!r}; choose from {sorted(PRESETS)}"
        )
    return MockProvider(simulator=Simulator.from_scenario(scenario, seed=settings.sim_seed))


def _auth(
    settings: Settings, config: RelayConfig, key_store: KeyStore | None
) -> tuple[Tenant | None, Authenticator | None]:
    """Either a fixed tenant (auth disabled, dev/test only) or an authenticator. Never both."""
    if settings.auth_disabled:
        if settings.env not in ("dev", "test"):
            # A config mistake that would expose every tenant's budget: refuse to start.
            raise ValueError("RELAY_AUTH_DISABLED is only allowed when RELAY_ENV is dev or test")
        tenant = config.tenants.get(settings.auth_disabled_tenant)
        if tenant is None:
            raise ValueError(
                f"RELAY_AUTH_DISABLED_TENANT {settings.auth_disabled_tenant!r} is not in the config"
            )
        logger.warning("auth is DISABLED: every request is tenant %r", tenant.id)
        return tenant, None

    if settings.key_pepper is None:
        raise ValueError(
            "RELAY_KEY_PEPPER must be set when auth is on (for a quick local run without keys, "
            "set RELAY_AUTH_DISABLED=true)"
        )
    if key_store is None:
        if settings.database_url:
            key_store = PostgresKeyStore(settings.database_url)
        else:
            logger.warning("RELAY_DATABASE_URL not set: no API key can authenticate")
            key_store = MemoryKeyStore()
    store = key_store if isinstance(key_store, CachedKeyStore) else CachedKeyStore(key_store)
    return None, Authenticator(store, settings.key_pepper.get_secret_value(), config)


def _capacity_store(settings: Settings) -> CapacityStore:
    if settings.redis_url:
        return ResilientStore(RedisStore(settings.redis_url), replica_count=settings.replica_count)
    if settings.replica_count > 1:
        # Each replica would enforce the full limit on its own: N replicas, N times the limit.
        raise ValueError("RELAY_REPLICA_COUNT > 1 needs RELAY_REDIS_URL for shared limits")
    logger.warning("RELAY_REDIS_URL not set: admission state is in-process (one replica only)")
    return MemoryStore()
