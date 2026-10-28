"""FastEmbedder only reports ready after a warm-up inference (found on the real stack: a cold
first embedding exceeded the router's timeout and the first request fell back to strong)."""

from typing import Any

import pytest

from relay.cache.embed import FastEmbedder


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> Any:
        self.calls.append(list(texts))
        return iter([[1.0, 0.0]] * len(texts))


async def test_start_warms_the_model_up_before_it_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    embedder = FastEmbedder()
    monkeypatch.setattr(embedder, "_load", lambda: model)
    assert not embedder.ready
    await embedder.start()
    assert embedder.ready
    assert model.calls == [["warm up"]]  # exactly one throwaway inference, before ready


async def test_a_load_failure_leaves_it_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = FastEmbedder()

    def boom() -> Any:
        raise RuntimeError("no model file")

    monkeypatch.setattr(embedder, "_load", boom)
    with pytest.raises(RuntimeError):
        await embedder.start()
    assert not embedder.ready  # the router and cache both route around it
