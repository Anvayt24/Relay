# syntax=docker/dockerfile:1

# ---- Stage 1: build the virtualenv ---------------------------------------------------
# A separate build stage keeps uv, build caches, and build-only files out of the final image.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /bin/uv

# Precompile .pyc files for faster startup. Copy (not hardlink) files out of uv's cache,
# because the cache is a separate mount. Never download Python; use the image's.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first, from the lockfile only. This layer is rebuilt only when
# pyproject.toml or uv.lock change, not on every source edit, which keeps rebuilds fast.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Then install the project itself, non-editable, so the runtime stage needs no source tree.
# --reinstall-package: uv keys a local package on its pyproject.toml, not its sources, so without
# this a source-only change silently keeps the OLD code in the image (found the hard way).
COPY README.md LICENSE ./
COPY relay ./relay
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --reinstall-package relay

# Bake the embedding model into the image (semantic cache, PRD FR-6). Downloading ~130 MB at
# startup would make a fresh container slow to become useful and break offline deploys; Relay
# also tolerates the model being absent (the semantic layer stays off), so this is an
# optimization, not a requirement.
ENV FASTEMBED_CACHE_PATH=/app/models
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# ---- Stage 2: runtime -----------------------------------------------------------------
# Same base image as the builder, so the venv's interpreter symlinks stay valid.
FROM python:3.12-slim AS runtime

# Run as an unprivileged user: a compromised process can't write to system paths.
RUN useradd --create-home --uid 10001 relay
WORKDIR /app
COPY --from=builder --chown=relay:relay /app/.venv /app/.venv
COPY --from=builder --chown=relay:relay /app/models /app/models

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models

USER relay
EXPOSE 8000
# --timeout-graceful-shutdown: on SIGTERM uvicorn waits this long for in-flight requests and
# streams to finish before the app's own shutdown drain runs (PRD §8).
CMD ["uvicorn", "relay.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
