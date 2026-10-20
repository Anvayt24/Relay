import asyncio
import time
from collections.abc import Sequence
from datetime import UTC, datetime

from prometheus_client import REGISTRY

from relay.telemetry.log_writer import LogRow, LogWriter, MemorySink


def _row(n: int) -> LogRow:
    return LogRow(
        request_id=f"req-{n}",
        ts=datetime.now(UTC),
        tenant="t",
        route="default",
        priority="interactive",
        cohort=None,
        stream=False,
        provider="mock",
        model="mock/strong",
        tier="strong",
        status="success",
        error_class=None,
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        ttft_ms=None,
    )


def _dropped(reason: str) -> float:
    return REGISTRY.get_sample_value("relay_log_dropped_total", {"reason": reason}) or 0.0


class RecordingSink:
    """Remembers each write() call's batch size."""

    def __init__(self) -> None:
        self.batches: list[int] = []
        self.rows: list[LogRow] = []

    async def write(self, rows: Sequence[LogRow]) -> None:
        self.batches.append(len(rows))
        self.rows.extend(rows)


class FailingSink:
    def __init__(self) -> None:
        self.attempts = 0

    async def write(self, rows: Sequence[LogRow]) -> None:
        self.attempts += 1
        raise ConnectionRefusedError("postgres is down")


class HangingSink:
    async def write(self, rows: Sequence[LogRow]) -> None:
        await asyncio.sleep(3600)


async def test_flushes_when_batch_is_full() -> None:
    sink = RecordingSink()
    writer = LogWriter(sink, batch_max=3, flush_interval_s=10)
    writer.start()
    for n in range(7):
        writer.enqueue(_row(n))
    await asyncio.sleep(0.05)
    # Two full batches went out immediately; the 7th row waits for more rows or the timer.
    assert sink.batches == [3, 3]
    await writer.stop(timeout_s=1)
    assert sink.batches == [3, 3, 1]
    assert [r.request_id for r in sink.rows] == [f"req-{n}" for n in range(7)]


async def test_flushes_partial_batch_after_interval() -> None:
    sink = RecordingSink()
    writer = LogWriter(sink, batch_max=100, flush_interval_s=0.05)
    writer.start()
    writer.enqueue(_row(1))
    writer.enqueue(_row(2))
    await asyncio.sleep(0.2)
    assert sink.batches == [2]
    await writer.stop(timeout_s=1)


async def test_full_queue_drops_the_oldest_row() -> None:
    sink = RecordingSink()
    writer = LogWriter(sink, max_queue=3)  # not started: nothing drains the queue
    before = _dropped("queue_full")

    for n in range(5):
        writer.enqueue(_row(n))

    assert _dropped("queue_full") == before + 2
    writer.start()
    await writer.stop(timeout_s=1)
    # The newest rows survive; they describe what is happening *now*.
    assert [r.request_id for r in sink.rows] == ["req-2", "req-3", "req-4"]


async def test_enqueue_never_blocks_even_when_the_sink_is_down() -> None:
    writer = LogWriter(FailingSink(), max_queue=100, backoff_base_s=0.01)
    writer.start()
    started = time.perf_counter()
    for n in range(10_000):
        writer.enqueue(_row(n))
    elapsed = time.perf_counter() - started
    # Synchronous, no awaits: thousands of rows in well under a second, whatever the sink does.
    assert elapsed < 1.0
    await writer.stop(timeout_s=0.2)


async def test_failing_sink_is_retried_then_the_batch_is_dropped() -> None:
    sink = FailingSink()
    writer = LogWriter(sink, max_attempts=3, backoff_base_s=0.001, flush_interval_s=0.01)
    before = _dropped("sink_failed")
    writer.start()
    writer.enqueue(_row(1))
    writer.enqueue(_row(2))
    await asyncio.sleep(0.2)

    assert sink.attempts == 3
    assert _dropped("sink_failed") == before + 2
    assert writer.last_write_ok is False
    await writer.stop(timeout_s=1)


async def test_recovers_after_sink_comes_back() -> None:
    class FlakySink(RecordingSink):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = 2

        async def write(self, rows: Sequence[LogRow]) -> None:
            if self.fail_next:
                self.fail_next -= 1
                raise ConnectionRefusedError
            await super().write(rows)

    sink = FlakySink()
    writer = LogWriter(sink, max_attempts=5, backoff_base_s=0.001, flush_interval_s=0.01)
    writer.start()
    writer.enqueue(_row(1))
    await asyncio.sleep(0.2)
    assert [r.request_id for r in sink.rows] == ["req-1"]
    assert writer.last_write_ok is True
    await writer.stop(timeout_s=1)


async def test_stop_flushes_everything_queued() -> None:
    sink = MemorySink()
    writer = LogWriter(sink, batch_max=10, flush_interval_s=10)
    writer.start()
    for n in range(25):
        writer.enqueue(_row(n))
    await writer.stop(timeout_s=1)
    assert len(sink.rows) == 25


async def test_stop_is_bounded_by_its_timeout() -> None:
    writer = LogWriter(HangingSink(), flush_interval_s=0.01, write_timeout_s=60)
    before = _dropped("shutdown")
    writer.start()
    writer.enqueue(_row(1))
    await asyncio.sleep(0.05)  # now stuck inside the hanging write

    started = time.perf_counter()
    await writer.stop(timeout_s=0.1)

    # A hung database must not hold up a deploy.
    assert time.perf_counter() - started < 1.0
    assert _dropped("shutdown") == before + 1
