# Relay task runner.
#
# Every Python recipe goes through `uv run`, which finds .venv itself and keeps it in sync
# with uv.lock, so the same recipes work in PowerShell (cmd.exe), Git Bash, and Linux CI.
# Keep recipes to plain commands: no shell-specific syntax (no &&, quotes, or pipes).

COMPOSE := docker compose -f deploy/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test test-integration migrate check up down logs \
        chaos load eval eval-router eval-cache eval-evidence reproduce

help:
	@echo Targets:
	@echo   install      create .venv and install exact versions from uv.lock
	@echo   fmt          auto-format and auto-fix lint issues
	@echo   check        lint + typecheck + test (what CI runs)
	@echo   lint / typecheck / test
	@echo   test-integration   tests that need Postgres (run make up first)
	@echo   migrate      apply SQL migrations to RELAY_DATABASE_URL
	@echo   up / down / logs   docker-compose stack
	@echo   chaos load eval eval-router eval-cache eval-evidence reproduce   (later phases)

install:
	uv sync --locked

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy relay

test:
	uv run pytest --cov --cov-report=term-missing

# The last -m wins over the one in pyproject's addopts, so this selects only integration tests.
test-integration:
	uv run pytest -m integration

migrate:
	uv run python -m relay.telemetry.migrate

check: lint typecheck test

# --wait blocks until every service with a healthcheck reports healthy.
up:
	$(COMPOSE) up -d --build --wait

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

# Stubs fail loudly: a reproduce script that "passes" without doing anything would be worse
# than one that is missing. Each is replaced in the phase listed in docs/plans/README.md.
chaos load eval eval-router eval-cache eval-evidence reproduce:
	@echo make $@: not implemented yet - see docs/plans/README.md for the phase that adds it
	@exit 1
