# CLAUDE.md

## Project
Relay: OpenAI-compatible LLM gateway with reliability, cost control, and
evidence-grade routing decisions. Read docs/PRD.md before any change.

## How we work
- Build in small phases (roadmap: docs/plans/README.md). Each phase has a plan
  (docs/plans/phase-NN.md) before coding and a learning guide (docs/learn/phase-NN.md) after.
- After each phase, STOP. The user reads the code to learn it in depth; do not start the next
  phase until they say so.
- Code is written to be read and learned from: explicit over clever, and comments explain
  *why*, not *what*.

## Commands
- `uv sync`: create/update .venv from uv.lock. Always use .venv, via `uv run ...`.
- `make check`: lint + typecheck + tests (what CI runs). `make fmt` auto-fixes.
- `make up` / `make down`: the docker-compose stack in deploy/.
- Dev machine is Windows: make is GNU make from winget, and recipes are plain `uv run`
  commands so they work in cmd, PowerShell, and Git Bash. Linux CI is authoritative.

## Rules
- Never call real provider APIs in tests; use the mock provider or simulator.
  If a real-provider call seems necessary, ask the user first.
  Real-provider tests are marked @pytest.mark.live and excluded from default CI.
- Logging is never on the hot path (async queue only).
- All pipeline stages implement the Stage protocol in PRD §9.3.
- Budget/admission/reservation code requires property-based tests.
- Any change to router, cache, or policy: re-run `make eval` and commit updated metrics.json.
- Write an ADR in docs/adr/ for each decision listed in PRD §15 before implementing it.
- Only cite numbers in README/docs that `make reproduce` produces.
- No silent model switching after the first byte has been streamed.
- Do not commit anything before asking the user. Never push without asking.

## Style
Python 3.12, type hints everywhere, mypy strict on relay/, ruff, Pydantic v2,
asyncio throughout, no blocking calls in the request path.
