"""The async request-log writer (PRD FR-8: "logging is never on the hot path").

The request path only ever calls ``enqueue(row)``. That is a plain *synchronous* method: it
puts the row on an in-memory queue and returns in microseconds. A background task drains the
queue in batches and writes them to a ``LogSink`` (Postgres in production).

Three design choices, each a classic trade-off:

1. **Bounded queue, drop oldest when full.** Unbounded, a Postgres outage would grow memory
   until the process dies. Blocking the request until there's room would make a logging
   outage a gateway outage. Dropping the *newest* would keep stale rows and lose the ones
   that describe the incident in progress. So when full, the oldest row is dropped and
   counted (``relay_log_dropped_total``).
2. **Batching** (up to ``batch_max`` rows or ``flush_interval_s``, whichever comes first).
   One ``COPY`` of 500 rows costs about the same as one ``INSERT`` of 1 row. Bigger batches
   mean more throughput, but rows wait longer before they are visible and more are at risk
   in memory if the process crashes.
3. **At-least-once delivery.** A failed write is retried with backoff. If a write committed
   but the acknowledgement was lost, the retry writes the batch again, so duplicates are
   possible and analytics should dedupe on ``request_id``. A batch that still fails after
   ``max_attempts`` is dropped and counted: memory stays bounded however long Postgres is
   down.
"""

import asyncio
import logging
import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)

DROPPED = Counter(
    "relay_log_dropped_total",
    "Request log rows that were never written.",
    ["reason"],  # queue_full | sink_failed | shutdown: a fixed set, so the label is bounded
)
WRITTEN = Counter("relay_log_rows_written_total", "Request log rows written to the sink.")
QUEUE_DEPTH = Gauge("relay_log_queue_depth", "Request log rows waiting to be written.")


@dataclass(frozen=True)
class LogRow:
    """One row of the ``requests`` table (a subset of PRD §9.4; later phases add columns).

    Field order is the column order used by the Postgres COPY.
    """

    request_id: str
    ts: datetime  # when the request arrived (UTC)
    tenant: str
    route: str
    priority: str
    cohort: str | None
    stream: bool
    provider: str | None  # None when the request never reached a provider
    model: str | None
    tier: str | None
    status: str  # success | error | cancelled | failed_midstream
    error_class: str | None
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int
    ttft_ms: int | None


class LogSink(Protocol):
    async def write(self, rows: Sequence[LogRow]) -> None: ...


class MemorySink:
    """Keeps rows in memory. Used by tests, and when no database is configured (then the
    newest ``maxlen`` rows are kept, so local dev never grows without bound)."""

    def __init__(self, maxlen: int = 10_000) -> None:
        self.rows: deque[LogRow] = deque(maxlen=maxlen)

    async def write(self, rows: Sequence[LogRow]) -> None:
        self.rows.extend(rows)


class LogWriter:
    def __init__(
        self,
        sink: LogSink,
        *,
        max_queue: int = 10_000,
        batch_max: int = 500,
        flush_interval_s: float = 0.5,
        max_attempts: int = 5,
        backoff_base_s: float = 0.1,
        backoff_max_s: float = 5.0,
        write_timeout_s: float = 5.0,
    ) -> None:
        self._sink = sink
        # Creating a Queue outside a running event loop is fine (Python 3.10+): it binds to
        # the loop lazily, on first use.
        self._queue: asyncio.Queue[LogRow] = asyncio.Queue(maxsize=max_queue)
        self._batch_max = batch_max
        self._flush_interval_s = flush_interval_s
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._write_timeout_s = write_timeout_s

        # The batch being collected or written. It is an attribute rather than a local so
        # that stop() can cancel the task while it waits for rows without losing the ones it
        # already took off the queue.
        self._batch: list[LogRow] = []
        self._task: asyncio.Task[None] | None = None
        self._writing = False
        self._stopping = False
        # None = nothing written yet. /readyz reports this as the sink's health.
        self.last_write_ok: bool | None = None

    # ---------------------------------------------------------------- request path
    def enqueue(self, row: LogRow) -> None:
        """Never blocks, never raises, never awaits."""
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            # No lock needed between these two calls: asyncio runs one coroutine at a time
            # and there is no `await` here, so nothing else can touch the queue in between.
            self._queue.get_nowait()
            self._queue.put_nowait(row)
            DROPPED.labels(reason="queue_full").inc()
        QUEUE_DEPTH.set(self._queue.qsize())

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="relay-log-writer")

    async def stop(self, timeout_s: float) -> None:
        """Flush what is queued, but never take longer than ``timeout_s``: a hung database
        must not stop a deploy. Whatever is left is counted as dropped."""
        self._stopping = True
        task = self._task
        try:
            async with asyncio.timeout(timeout_s):
                if task is not None:
                    if not self._writing:
                        # It is waiting for rows, so cancelling is safe: rows it already took
                        # are in self._batch. If it is mid-write, let that write finish.
                        task.cancel()
                    # asyncio.wait(), not `with suppress(CancelledError): await task`. That
                    # suppress would also swallow the CancelledError that asyncio.timeout uses
                    # to interrupt *us*, silently disabling the timeout. wait() returns when
                    # the task is done, without re-raising the task's own cancellation.
                    await asyncio.wait([task])
                while self._batch or not self._queue.empty():
                    self._take_available()
                    await self._flush()
        except TimeoutError:
            lost = len(self._batch) + self._queue.qsize()
            DROPPED.labels(reason="shutdown").inc(lost)
            logger.warning("log writer stopped with %d unwritten rows", lost)
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.wait([task])

    # ---------------------------------------------------------------- background task
    async def _run(self) -> None:
        while not self._stopping:
            await self._collect()
            await self._flush()

    async def _collect(self) -> None:
        """Fill self._batch: wait (without spinning) for the first row, then take more until
        the batch is full or flush_interval_s has passed since that first row."""
        if not self._batch:
            self._batch.append(await self._queue.get())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._flush_interval_s
        while len(self._batch) < self._batch_max:
            self._take_available()
            remaining = deadline - loop.time()
            if len(self._batch) >= self._batch_max or remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
                    self._batch.append(await self._queue.get())
            except TimeoutError:
                break

    def _take_available(self) -> None:
        while len(self._batch) < self._batch_max and not self._queue.empty():
            self._batch.append(self._queue.get_nowait())
        QUEUE_DEPTH.set(self._queue.qsize())

    async def _flush(self) -> None:
        if not self._batch:
            return
        self._writing = True
        try:
            await self._write_with_retry(self._batch)
            # Cleared only once the write is settled (written, or dropped and counted). Not in
            # `finally`: if shutdown cancels this mid-write, the rows must still be there to
            # be counted as lost.
            self._batch = []
        finally:
            self._writing = False

    async def _write_with_retry(self, batch: Sequence[LogRow]) -> None:
        for attempt in range(1, self._max_attempts + 1):
            try:
                async with asyncio.timeout(self._write_timeout_s):
                    await self._sink.write(batch)
            except Exception as e:
                self.last_write_ok = False
                if attempt == self._max_attempts:
                    DROPPED.labels(reason="sink_failed").inc(len(batch))
                    logger.warning(
                        "dropping %d log rows after %d attempts: %r", len(batch), attempt, e
                    )
                    return
                await asyncio.sleep(self._backoff(attempt))
            else:
                self.last_write_ok = True
                WRITTEN.inc(len(batch))
                return

    def _backoff(self, attempt: int) -> float:
        # "Full jitter": a random wait between 0 and the exponential cap. Without jitter,
        # every replica that lost the database at the same moment retries at the same
        # moments too, so the recovering database gets hit by synchronized waves.
        cap = min(self._backoff_max_s, self._backoff_base_s * 2 ** (attempt - 1))
        return random.uniform(0, cap)
