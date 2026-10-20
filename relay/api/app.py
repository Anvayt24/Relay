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
from relay.api import health, routes
from relay.api.errors import install_error_handlers
from relay.api.request_id import RequestIdMiddleware
from relay.config.models import ALIASES, MODELS
from relay.pipeline.runner import Pipeline
from relay.pipeline.stages import LogStage, ProviderStage, RouteStage
from relay.providers.base import ProviderAdapter
from relay.providers.mock import MockProvider
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
) -> FastAPI:
    settings = settings or Settings()
    if providers is None:
        providers = {"mock": _mock_provider(settings)}

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

    # Built once per app, not per request. Constructors validate the config (dangling
    # aliases, models without an adapter), so a bad config fails at startup.
    pipeline = Pipeline(
        [
            LogStage(writer),  # first in, so its finalize is last out and sees the outcome
            RouteStage(StaticRouter(MODELS, ALIASES)),
            ProviderStage(MODELS, providers, first_token_timeout_s=settings.first_token_timeout_s),
        ]
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if postgres is not None:
            await postgres.start()  # creates the pool; connects to nothing yet
        elif memory_fallback:
            logger.warning("RELAY_DATABASE_URL not set: request logs are kept in memory only")
        writer.start()
        try:
            yield
        finally:
            await writer.stop(settings.log_shutdown_timeout_s)
            if postgres is not None:
                await postgres.close()

    app = FastAPI(title="Relay", version=__version__, lifespan=lifespan)
    # app.state is how request handlers reach process-wide objects without globals.
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.models = MODELS
    app.state.aliases = ALIASES
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
