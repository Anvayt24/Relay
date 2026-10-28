"""Prompt embeddings, computed locally and off the event loop (PRD FR-6, FR-7; ADR 0016).

An embedding turns a prompt into a vector so that *similar prompts are near each other*. The
semantic cache looks for a stored prompt close enough to the new one; the router (Phase 11)
uses the same vector as a feature. It is computed **once per request** and kept on the
context, so the router doesn't pay for it again.

Inference is CPU-bound, so it runs in a thread pool: running it on the event loop would stall
every other request for the milliseconds it takes, which is exactly what "no blocking calls in
the request path" forbids. (ONNX Runtime releases the GIL during inference, so the threads
really run in parallel with the loop.)

Two implementations:
- ``FastEmbedder``: ``BAAI/bge-small-en-v1.5`` through fastembed (ONNX), L2-normalized.
- ``HashEmbedder``: a deterministic fake for tests: hashes word n-grams into a fixed-size
  vector. Similar texts share n-grams and so end up near each other, which is enough to
  exercise the cache logic without a model download.

Vectors are L2-normalized, so cosine similarity is a dot product.
"""

import asyncio
import hashlib
import itertools
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384


class Embedder(Protocol):
    name: str
    dimensions: int

    @property
    def ready(self) -> bool: ...

    async def embed(self, text: str) -> list[float]: ...


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class HashEmbedder:
    """Feature hashing of word unigrams and bigrams. Not semantic, but similar word sets
    give similar vectors, which is all the cache tests need."""

    name = "hash-embedder"
    ready = True

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        words = text.lower().split()
        grams = words + [f"{a} {b}" for a, b in itertools.pairwise(words)]
        vec = [0.0] * self.dimensions
        for gram in grams:
            h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
            vec[h % self.dimensions] += 1.0 if (h >> 63) & 1 else -1.0
        return normalize(vec)


class FastEmbedder:
    name = MODEL_NAME
    dimensions = DIMENSIONS

    def __init__(
        self, model_name: str = MODEL_NAME, *, threads: int = 2, onnx_threads: int = 1
    ) -> None:
        # `threads`: how many embeddings may run at once (the executor). `onnx_threads`: how many
        # threads ONNX Runtime uses *inside* one embedding. It defaults to ALL visible cores,
        # and in a container limited to 2 CPUs on a 12-core host that means 12 threads fighting
        # over 2: measured 121 ms of CPU per short prompt, against 12 ms with one thread. Keep
        # it at 1 and scale with `threads` / more worker processes instead.
        self._model_name = model_name
        self._onnx_threads = onnx_threads
        self._model: Any = None  # loaded in start(); a failure there disables the cache
        self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="embed")

    @property
    def ready(self) -> bool:
        return self._model is not None

    async def start(self) -> None:
        """Load the model (downloads it on first use). Raises if it can't be loaded."""
        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(self._pool, self._load)
        # Warm up before declaring ready. The first ONNX inference is much slower than the rest
        # (graph optimization, buffer allocation): measured, it exceeded the router's 0.25 s
        # embedding timeout, so the first request after startup fell back to the strong model.
        # `ready` flips only after this, so nothing ever waits on a cold model.
        await loop.run_in_executor(self._pool, lambda: list(model.embed(["warm up"])))
        self._model = model

    def start_in_background(self) -> asyncio.Task[None]:
        """Load the model without delaying startup. Relay must come up fast (health checks
        time out) even if the model is slow to download or missing: until it is ready, the
        semantic layer is simply skipped, and a load failure leaves it off (degradation
        matrix: "embedding model fails to load: semantic cache disabled")."""

        async def load() -> None:
            try:
                await self.start()
                logger.info("embedding model %s loaded", self._model_name)
            except Exception:
                logger.exception("embedding model failed to load: semantic cache disabled")

        return asyncio.create_task(load(), name="relay-embedder-load")

    def _load(self) -> Any:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self._model_name, threads=self._onnx_threads)

    def _embed_sync(self, text: str) -> list[float]:
        assert self._model is not None
        vec = next(iter(self._model.embed([text])))
        return normalize([float(x) for x in vec])

    async def embed(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError("FastEmbedder.start() was not called")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._embed_sync, text)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding, for building the frozen evaluation data."""
        if self._model is None:
            raise RuntimeError("FastEmbedder.start() was not called")
        model = self._model

        def run() -> list[list[float]]:
            return [normalize([float(x) for x in v]) for v in model.embed(texts)]

        return await asyncio.get_running_loop().run_in_executor(self._pool, run)

    async def close(self) -> None:
        self._pool.shutdown(wait=False)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
