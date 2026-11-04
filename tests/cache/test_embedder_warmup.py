"""FastEmbedder only reports ready after a warm-up inference (found on the real stack: a cold
first embedding exceeded the router's timeout and the first request fell back to strong)."""

import asyncio
import threading
from typing import Any

import pytest

from relay.cache import embed
from relay.cache.embed import EmbedderBusy, FastEmbedder


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


class BlockingModel:
    """Each embedding blocks its thread until released, like a model on a starved CPU."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def embed(self, texts: list[str]) -> Any:
        self.release.wait(5)
        return iter([[1.0, 0.0]] * len(texts))


async def test_a_backlog_is_refused_at_once_and_drains_when_the_jobs_end() -> None:
    # Found in review: a timeout cancels the caller's await but not the job on the thread
    # pool, so a slow spell built a backlog in which every later embedding timed out too.
    model = BlockingModel()
    embedder = FastEmbedder()
    embedder._model = model
    waiting = [asyncio.ensure_future(embedder.embed("x")) for _ in range(embed.MAX_IN_FLIGHT)]
    await asyncio.sleep(0)
    with pytest.raises(EmbedderBusy):
        await embedder.embed("one too many")
    for task in waiting:  # the callers give up (their timeout)...
        task.cancel()
    await asyncio.sleep(0.05)
    # Queued jobs are cancelled before they start; the ones already running keep their
    # places until they really end.
    assert embedder._in_flight == embed.THREADS
    model.release.set()  # the jobs finish
    for _ in range(100):
        if embedder._in_flight == 0:
            break
        await asyncio.sleep(0.02)
    assert embedder._in_flight == 0
    assert await embedder.embed("now there is room") == [1.0, 0.0]
    await embedder.close()
