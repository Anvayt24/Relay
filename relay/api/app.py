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

Uvicorn runs lifespan shutdown only after in-flight requests have finished, streams
included. By the time the writer flushes, every request's log row is already queued.

Run it with ``uvicorn relay.api.app:create_app --factory``.
"""

import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager

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
from relay.auth.service import Authenticator
from relay.auth.store import CachedKeyStore, KeyStore, MemoryKeyStore, PostgresKeyStore
from relay.config.loader import load_config
from relay.config.models import available_models
from relay.config.schema import RelayConfig
from relay.config.tenants import Tenant
from relay.pipeline.runner import Pipeline
from relay.pipeline.stages import LogStage, ProviderStage, RouteStage
from relay.providers.base import ProviderAdapter
from relay.providers.mock import MockProvider
from relay.providers.registry import ProviderKeys, build_adapters
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
) -> FastAPI:
    settings = settings or Settings()
    # Invalid YAML (dangling alias, cyclic fallback graph, ...) raises here: Relay won't start.
    config = config or load_config(settings.config_path)
    if providers is None:
        # The mock provider is always there; real providers only when their key is set.
        keys = provider_keys if provider_keys is not None else ProviderKeys()
        providers = {
            "mock": _mock_provider(settings),
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
    pipeline = Pipeline(
        [
            LogStage(writer),  # first in, so its finalize is last out and sees the outcome
            AdmissionStage(store, queue, replica_count=settings.replica_count),
            RouteStage(StaticRouter(config, models), config),
            ReservationStage(store, config),  # after routing: the price needs the model
            ProviderStage(
                config,
                models,
                providers,
                first_token_timeout_s=settings.first_token_timeout_s,
                pause=PauseState(store, config),
            ),
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
        writer.start()
        try:
            yield
        finally:
            await writer.stop(settings.log_shutdown_timeout_s)
            if postgres is not None:
                await postgres.close()
            if authenticator is not None:
                await authenticator.store.close()
            await store.close()
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

    install_error_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    app.include_router(routes.router)
    return app


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
