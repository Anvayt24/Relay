# Relay

An OpenAI-compatible LLM gateway. It keeps apps up when model providers fail, cuts spend by sending easy requests to cheap models, and makes routing and fallback decisions from measured evidence.

> **Status:** Phase 0 (scaffold). Only `/healthz`, `/readyz` and `/metrics` exist so far.
> The roadmap is in [docs/plans/README.md](docs/plans/README.md) and the full spec in [docs/PRD.md](docs/PRD.md).

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/), GNU make, and Docker.

```sh
uv sync          # create .venv and install the exact versions pinned in uv.lock
make check       # lint + type-check + tests
make up          # start api, redis, postgres+pgvector, otel-collector, prometheus, grafana
curl localhost:8000/healthz
make down
```

| Service | URL |
|---|---|
| Relay API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

## Repository map

| Path | What lives there |
|---|---|
| `relay/` | The gateway. Each subpackage's `__init__.py` says what it owns |
| `tests/` | pytest suite (`make test`) |
| `bench/` | Provider simulator, chaos, and load tests |
| `evals/` | Router, cache, and evidence evaluations |
| `deploy/` | docker-compose and service configs |
| `docs/` | PRD, design doc, ADRs, per-phase plans, and learning guides |

## License

MIT
