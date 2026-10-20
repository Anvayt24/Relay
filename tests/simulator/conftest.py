from collections.abc import Callable
from typing import Any

import pytest

from relay.schemas.chat import ChatRequest


class FakeClock:
    """A clock that only moves when the test says so. Capacity and incident tests then check
    exact numbers ("retry after 0.5 s") without sleeping or flaking."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def _chat_request(
    model: str = "mock-strong-v1", content: str = "hello there", **extra: Any
) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    )


@pytest.fixture
def make_request() -> Callable[..., ChatRequest]:
    # A fixture, not an import: with --import-mode=importlib, test folders are not packages,
    # so tests can't import helpers from conftest.
    return _chat_request
