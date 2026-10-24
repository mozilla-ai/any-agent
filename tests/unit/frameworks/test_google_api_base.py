from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from any_agent import AgentConfig, AgentFramework, AnyAgent


@pytest.mark.parametrize(
    ("explicit", "legacy", "has_legacy", "expected"),
    [
        (None, None, False, None),
        (None, "https://legacy.example/v1", True, "https://legacy.example/v1"),
        ("https://explicit.example/v1", None, False, "https://explicit.example/v1"),
        (
            "https://explicit.example/v1",
            "https://legacy.example/v1",
            True,
            "https://explicit.example/v1",
        ),
        (None, None, True, None),
        ("https://explicit.example/v1", None, True, "https://explicit.example/v1"),
        ("", "https://legacy.example/v1", True, ""),
        (None, "", True, ""),
    ],
)
@pytest.mark.parametrize("structured", [False, True])
async def test_google_endpoint_reaches_completion_without_mutating_config(
    explicit: str | None,
    legacy: str | None,
    has_legacy: bool,
    expected: str | None,
    structured: bool,
) -> None:
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from any_agent.frameworks.google import AnyLlm, GoogleAgent

    class Result(BaseModel):
        answer: str

    model_args: dict[str, Any] = {"temperature": 0.25}
    if has_legacy:
        model_args["api_base"] = legacy
    original_args = dict(model_args)
    config = AgentConfig(
        model_id="openai:test-model",
        api_base=explicit,
        api_key="not-a-real-key",
        model_args=model_args,
        output_type=Result if structured else None,
        callbacks=[],
    )
    response = ChatCompletion(
        id="offline",
        created=0,
        model="test-model",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="ok"),
            )
        ],
    )
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="hello")])]
    )
    completion = AsyncMock(return_value=response)
    with patch("any_agent.frameworks.google.acompletion", completion):
        for _ in range(2):
            agent = await AnyAgent.create_async(AgentFramework.GOOGLE, config)
            assert isinstance(agent, GoogleAgent)
            assert agent._agent is not None
            model = agent._agent.model
            assert isinstance(model, AnyLlm)
            replies = [reply async for reply in model.generate_content_async(request)]
            content = replies[0].content
            assert content is not None
            assert content.parts is not None
            assert content.parts[0].text == "ok"
            call = completion.await_args
            assert call is not None
            sent = call.kwargs
            assert sent["api_base"] == expected
            assert sent["api_key"] == "not-a-real-key"
            assert sent["temperature"] == 0.25
            if structured:
                assert sent["tool_choice"] == "required"
            else:
                assert "tool_choice" not in sent
            assert config.model_args == original_args
            assert model_args == original_args
