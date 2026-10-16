"""Relay: an OpenAI-compatible LLM gateway.

The subpackages follow the request path described in docs/PRD.md §9:

    api -> pipeline (auth -> admission -> cache -> routing -> reservation -> providers)
                                                     -> telemetry (async logging)
    evidence and policy run beside the request path and feed decisions back into it.
"""

from importlib.metadata import version

# Read from the installed package metadata, so pyproject.toml is the single source of truth.
__version__ = version("relay")
