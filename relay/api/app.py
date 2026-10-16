"""FastAPI application factory.

Why a factory (``create_app()``) instead of a module-level ``app = FastAPI()``:
- Tests build a fresh app per test with their own settings and fakes. No global state
  leaks between tests.
- Importing this module has no side effects (no connections opened, no env read).
- Later phases inject dependencies (providers, log sinks, config) as arguments, which
  makes the wiring explicit in one place.

Run it with ``uvicorn relay.api.app:create_app --factory``.
"""

from fastapi import FastAPI

from relay import __version__
from relay.api import health
from relay.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    app = FastAPI(title="Relay", version=__version__)
    # app.state is how request handlers reach process-wide objects without globals.
    app.state.settings = settings
    app.include_router(health.router)
    return app
