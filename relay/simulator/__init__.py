"""The provider simulator (PRD §11.1): a fake LLM provider whose behavior is known exactly.

Reading order, from the pure core outwards:

- ``seeding``:  deterministic per-request randomness (seed, index, model, purpose).
- ``scenario``: what each simulated model is like (latency, length, errors, quality,
  capacity, incidents), plus the ``PRESETS``.
- ``quality``:  hidden correctness ``c ~ Bernoulli(q)``, and solving q from a target win rate.
- ``judge``:    a mock judge with tunable sensitivity and false-loss rate, and drift.
- ``capacity``: RPM/TPM token buckets per API key, and the injectable clock.
- ``plan``:     the per-request decision (pure); scripted and sampled planners.
- ``engine``:   ``Simulator`` gates and plans a request, and ``Call`` executes the plan.
- ``server``:   HTTP mode, speaking OpenAI's and Anthropic's wire formats.

Why this lives in ``relay/`` and not ``bench/simulator/`` (PRD §9.6): the PRD also makes the
simulator Relay's in-process ``mock`` provider, and the Docker image ships only the ``relay``
package. See ADR 0004.
"""
