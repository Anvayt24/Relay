"""FastAPI application factory.

Why a factory (``create_app()``) instead of a module-level ``app = FastAPI()``:
- Tests build a fresh app per test with their own settings and fakes. No global state
  leaks between tests.
- Importing this module has no side effects (no connections opened, no env read).
- Dependencies (providers now; log sinks and config later) are passed in as arguments,
  so all the wiring is explicit and in one place.

Run it with ``uvicorn relay.api.app:create_app --factory``.
"""

from collections.abc import Mapping

from fastapi import FastAPI

from relay import __version__
from relay.api import health, routes
from relay.api.errors import install_error_handlers
from relay.api.request_id import RequestIdMiddleware
from relay.config.models import ALIASES, MODELS
from relay.pipeline.runner import Pipeline
from relay.pipeline.stages import ProviderStage, RouteStage
from relay.providers.base import ProviderAdapter
from relay.providers.mock import MockProvider
from relay.routing.static import StaticRouter
from relay.settings import Settings


def create_app(
    settings: Settings | None = None,
    providers: Mapping[str, ProviderAdapter] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if providers is None:
        providers = {"mock": MockProvider()}

    # Built once per app, not per request. Constructors validate the config (dangling
    # aliases, models without an adapter), so a bad config fails at startup.
    pipeline = Pipeline(
        [
            RouteStage(StaticRouter(MODELS, ALIASES)),
            ProviderStage(MODELS, providers),
        ]
    )

    app = FastAPI(title="Relay", version=__version__)
    # app.state is how request handlers reach process-wide objects without globals.
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.models = MODELS
    app.state.aliases = ALIASES

    install_error_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(health.router)
    app.include_router(routes.router)
    return app
