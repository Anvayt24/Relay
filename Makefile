# Relay task runner.
#
# Every Python recipe goes through `uv run`, which finds .venv itself and keeps it in sync
# with uv.lock, so the same recipes work in PowerShell (cmd.exe), Git Bash, and Linux CI.
# Keep recipes to plain commands: no shell-specific syntax (no &&, quotes, or pipes).

COMPOSE := docker compose -f deploy/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test cov-gate test-integration migrate sim check up down logs \
        chaos chaos-smoke load eval eval-judge eval-router eval-cache eval-evidence reproduce \
        train-router

help:
	@echo Targets:
	@echo   install      create .venv and install exact versions from uv.lock
	@echo   fmt          auto-format and auto-fix lint issues
	@echo   check        lint + typecheck + test (what CI runs)
	@echo   lint / typecheck / test
	@echo   test-integration   tests that need Postgres (run make up first)
	@echo   migrate      apply SQL migrations to RELAY_DATABASE_URL
	@echo   sim          run the provider simulator on localhost:8100 (realistic preset)
	@echo   up / down / logs   docker-compose stack
	@echo   chaos-smoke  compressed fault scenarios asserting documented behavior (localhost)
	@echo   chaos        the full chaos suite, 5 runs per scenario and client -> bench/results/chaos
	@echo   load         k6 load test, 1 vs 2 replicas, in Docker -> bench/results/load
	@echo   eval          compare every eval against its committed metrics.json (part of check)
	@echo   eval-cache   re-run the cache eval and rewrite evals/cache/metrics.json
	@echo   eval-judge   validate the judge calibration against the simulator
	@echo   train-router re-train the router (sklearn) and rewrite its artifact
	@echo   eval-router  re-run the router eval and rewrite evals/router/sim/metrics.json
	@echo   eval-evidence reproduce   (later phases)

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

# PRD NFR: >= 80% line coverage on the packages that carry the guarantees. Reads the data
# file `make test` just wrote.
cov-gate:
	uv run coverage report --include=relay/pipeline/*,relay/reliability/*,relay/evidence/*,relay/routing/* --fail-under=80

# The last -m wins over the one in pyproject's addopts, so this selects only integration tests.
test-integration:
	uv run pytest -m integration

migrate:
	uv run python -m relay.telemetry.migrate

# The provider simulator over HTTP (OpenAI + Anthropic wire). Options: --scenario, --seed, --port.
sim:
	uv run python -m relay.simulator

check: lint typecheck test cov-gate eval

# --wait blocks until every service with a healthcheck reports healthy.
up:
	$(COMPOSE) up -d --build --wait

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

# Chaos (PRD E-C). Everything is localhost subprocesses: simulators and Relay, no network.
chaos-smoke:
	uv run pytest -m chaos tests/bench

chaos:
	uv run python -m bench.chaos --runs 5

# Load (PRD E-D, M1). Needs Docker: k6 runs in a pinned image, the stack in compose.
load:
	uv run python -m bench.load

# Evals (CLAUDE.md: any change to router, cache or policy re-runs `make eval` and commits the
# updated metrics.json). `eval` only compares, so CI fails on a regression; `eval-cache`
# rewrites the baseline. The cache eval reads frozen embeddings: no model, no network.
eval:
	uv run python -m evals.cache.evaluate --check
	uv run python -m evals.router.evaluate --check
	uv run python -m evals.router.evaluate --world real --check

eval-cache:
	uv run python -m evals.cache.evaluate

# Judge calibration machinery vs a simulated judge of known accuracy (Phase 9).
eval-judge:
	uv run python -m relay.evidence.validate --out evals/judge/sim_validation.json

# Stubs fail loudly: a reproduce script that "passes" without doing anything would be worse
# than one that is missing. Each is replaced in the phase listed in docs/plans/README.md.
# The router (PRD FR-7). train-router fits it with scikit-learn and writes the artifact;
# eval-router measures the *committed* artifact on the frozen data (no model, no network).
# Rebuild the dataset only to change it: `python -m evals.router.build` (needs the embedding model).
train-router:
	uv run python -m evals.router.train

eval-router:
	uv run python -m evals.router.evaluate

eval-evidence reproduce:
	@echo make $@: not implemented yet - see docs/plans/README.md for the phase that adds it
	@exit 1
