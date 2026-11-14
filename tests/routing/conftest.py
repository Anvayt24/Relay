"""Helpers for router tests: a tiny hand-built artifact whose behavior is obvious.

The artifact looks only at prompt length: short prompts score high (easy, send cheap), long
ones score low (send strong). With zero embedding weights and an identity scaler,

    logit = BIAS - SLOPE * log1p(len(text))

so a 10-character prompt scores about 0.95 and a 400-character one about 0.03, and the
embedding never matters. That lets router tests assert exact decisions without a real model.
"""

import asyncio
from typing import Any

import numpy as np
import pytest

from relay.cache.embed import DIMENSIONS, HashEmbedder
from relay.config.schema import RouterSpec
from relay.routing.artifact import RouterArtifact
from relay.routing.features import N_SCALARS, SCALAR_NAMES
from relay.schemas.chat import ChatRequest

from ..conftest import TEST_CONFIG  # type: ignore[import-not-found]

BIAS, SLOPE = 5.0, 1.0


def length_artifact(
    *, embedding_model: str = "hash-embedder", dimensions: int = DIMENSIONS, version: str = "t-v1"
) -> RouterArtifact:
    weights = np.zeros(dimensions + N_SCALARS)
    weights[dimensions + SCALAR_NAMES.index("log_chars")] = -SLOPE
    return RouterArtifact(
        version=version,
        embedding_model=embedding_model,
        dimensions=dimensions,
        weights=weights,
        bias=BIAS,
        scaler_mean=np.zeros(N_SCALARS),
        scaler_scale=np.ones(N_SCALARS),
        threshold=0.5,
    )


def chat(content: str, model: str = "relay/auto", **extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    )


class StubEmbedder(HashEmbedder):
    """A HashEmbedder that counts calls and can be made slow, broken, or not ready."""

    def __init__(
        self,
        *,
        ready: bool = True,
        delay_s: float = 0.0,
        fail: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._ready = ready
        self._delay_s = delay_s
        self._fail = fail
        if name is not None:
            self.name = name
        self.calls = 0

    @property  # type: ignore[override]
    def ready(self) -> bool:
        return self._ready

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._fail:
            raise RuntimeError("model blew up")
        return await super().embed(text)


def routed_config(threshold: float = 0.5, **spec: Any) -> Any:
    """TEST_CONFIG with a router on the default route (aliases: cheap/strong/auto)."""
    route = TEST_CONFIG.routes["default"].model_copy(
        update={"router": RouterSpec(artifact="unused.json", threshold=threshold, **spec)}
    )
    return TEST_CONFIG.model_copy(update={"routes": {**TEST_CONFIG.routes, "default": route}})


@pytest.fixture
def config() -> Any:
    return routed_config()
