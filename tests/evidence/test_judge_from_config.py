"""The app builds each route's judge from its evidence section: rubric version and temperature."""

import copy
from typing import Any

from relay.api.app import _judge_factory
from relay.config.loader import load_config
from relay.config.models import available_models
from relay.config.schema import RelayConfig
from relay.schemas.chat import ChatRequest, ChatResponse, Choice, ResponseMessage, Usage


class RecordingAdapter:
    name = "mock"

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        message = ResponseMessage(content='{"reason": "r", "verdict": "neither"}')
        return ChatResponse(
            id="x",
            created=0,
            model=req.model,
            choices=[Choice(index=0, message=message, finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def _config(**evidence: Any) -> RelayConfig:
    shipped = load_config()
    data: dict[str, Any] = copy.deepcopy(shipped.model_dump(mode="json"))
    data["routes"]["default"]["evidence"] = {"judge": "mock/cheap", **evidence}
    return RelayConfig.model_validate(data)


async def test_the_judge_asks_for_the_routes_rubric_and_temperature() -> None:
    config = _config(judge_prompt_version="v2", judge_temperature=None, judge_max_tokens=600)
    adapter = RecordingAdapter()
    models = available_models(config.models, {"mock": adapter})
    judge = _judge_factory({"mock": adapter}, models, config)("mock/cheap", 600)  # type: ignore[dict-item]
    assert judge is not None
    result = await judge.judge(request="q", candidate="c", reference="r")
    sent = adapter.requests[0]
    assert sent.temperature is None  # Sonnet 5.5 refuses any other value
    assert "reason" in sent.messages[0].content  # rubric v2 asks for the reason first
    assert result.prompt_version == "v2" and result.reason == "r"


async def test_a_routes_default_judge_is_rubric_v1_at_temperature_zero() -> None:
    config = _config()
    adapter = RecordingAdapter()
    models = available_models(config.models, {"mock": adapter})
    judge = _judge_factory({"mock": adapter}, models, config)("mock/cheap", 16)  # type: ignore[dict-item]
    assert judge is not None
    result = await judge.judge(request="q", candidate="c", reference="r")
    assert adapter.requests[0].temperature == 0
    assert result.prompt_version == "v1"
