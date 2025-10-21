from typing import TYPE_CHECKING
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall
from langchain_core.tools import tool
from pydantic import BaseModel

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.frameworks.langchain import ChatAnyLLM
from any_agent.vendor.langchain_any_llm import _convert_message_to_dict

if TYPE_CHECKING:
    from any_agent.frameworks.langchain import LangchainAgent


def test_load_langchain_agent_default() -> None:
    model_mock = MagicMock()
    create_mock = MagicMock()
    agent_mock = MagicMock()
    create_mock.return_value = agent_mock
    tool_mock = MagicMock()

    with (
        patch("any_agent.frameworks.langchain.DEFAULT_AGENT_TYPE", create_mock),
        patch("any_agent.frameworks.langchain.DEFAULT_MODEL_TYPE", model_mock),
        patch("langchain_core.tools.tool", tool_mock),
    ):
        AnyAgent.create(
            AgentFramework.LANGCHAIN,
            AgentConfig(model_id="mistral:mistral-small-latest"),
        )

        model_mock.assert_called_once_with(
            model="mistral:mistral-small-latest",
            api_base=None,
            api_key=None,
            model_kwargs={},
        )
        create_mock.assert_called_once_with(
            name="any_agent",
            model=model_mock.return_value,
            tools=[],
            prompt=None,
        )


def test_load_langchain_agent_missing() -> None:
    with patch("any_agent.frameworks.langchain.langchain_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(
                AgentFramework.LANGCHAIN,
                AgentConfig(model_id="mistral:mistral-small-latest"),
            )


def test_run_langchain_agent_custom_args() -> None:
    create_mock = MagicMock()
    agent_mock = AsyncMock()
    agent_mock.ainvoke.return_value = MagicMock()
    create_mock.return_value = agent_mock

    with (
        patch("any_agent.frameworks.langchain.DEFAULT_AGENT_TYPE", create_mock),
        patch("any_agent.frameworks.langchain.DEFAULT_MODEL_TYPE"),
        patch("langchain_core.tools.tool"),
    ):
        agent = AnyAgent.create(
            AgentFramework.LANGCHAIN,
            AgentConfig(model_id="mistral:mistral-small-latest"),
        )
        agent.run("foo", debug=True)
        agent_mock.ainvoke.assert_called_once_with(
            {"messages": [("user", "foo")]}, debug=True, config={"callbacks": [ANY]}
        )


class SampleOutput(BaseModel):
    answer: str
    confidence: float


def test_structured_output_without_tools() -> None:
    """Test that structured output works correctly when no tools are present and tool-related params are not set."""
    config = AgentConfig(model_id="openai:gpt-4o-mini", output_type=SampleOutput)
    agent: LangchainAgent = AnyAgent.create(AgentFramework.LANGCHAIN, config)  # type: ignore[assignment]

    # Patch the agent's _agent to return a mock result for ainvoke
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [AIMessage(content="Initial response")]
    }
    agent._agent = mock_agent

    def create_mock_response(content: str) -> ChatCompletion:
        return ChatCompletion(
            id="chatcmpl-test",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content=content, role="assistant"),
                )
            ],
            created=1747157127,
            model="gpt-4o-mini",
            object="chat.completion",
        )

    with patch("any_agent.frameworks.langchain.acompletion") as mock_acompletion:
        mock_acompletion.return_value = create_mock_response(
            '{"answer": "Structured answer", "confidence": 0.95}'
        )
        agent.run("Test question")

        # Only expect that acompletion was called once for structured output
        assert mock_acompletion.call_count == 1
        call_args = mock_acompletion.call_args[1]
        # Should not include any tool-related keys
        assert "tools" not in call_args
        assert "tool_choice" not in call_args
        # Should include response_format
        assert "response_format" in call_args
        assert call_args["response_format"] == SampleOutput


def test_chat_anyllm_create_message_dicts() -> None:
    """Test that ChatAnyLLM properly converts Langchain messages to API format with correct parameters."""
    chat_model = ChatAnyLLM(
        model="gpt-4o",
        api_key="test-key",
        api_base="https://test.com",
        model_kwargs={"temperature": 0.7, "max_tokens": 100},
    )

    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Hello"),
        AIMessage(
            content="Hi there!",
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", args={"location": "NYC"})
            ],
        ),
    ]

    message_dicts = [_convert_message_to_dict(m) for m in messages]
    params = chat_model._create_params(stop=["END"])

    assert len(message_dicts) == 3
    assert message_dicts[0] == {
        "role": "system",
        "content": "You are a helpful assistant",
    }
    assert message_dicts[1] == {"role": "user", "content": "Hello"}
    assert message_dicts[2]["role"] == "assistant"
    assert message_dicts[2]["content"] == "Hi there!"
    assert "tool_calls" in message_dicts[2]
    assert message_dicts[2]["tool_calls"][0]["function"]["name"] == "get_weather"

    assert params["model"] == "gpt-4o"
    assert params["api_key"] == "test-key"
    assert params["api_base"] == "https://test.com"
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 100
    assert params["stop"] == ["END"]


def test_chat_anyllm_create_chat_result() -> None:
    """Test that ChatAnyLLM properly converts API responses to Langchain format with token usage."""
    chat_model = ChatAnyLLM(
        model="gpt-4o",
        api_key=None,
        api_base=None,
        model_kwargs={},
    )

    response = ChatCompletion(
        id="chatcmpl-123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="This is a test response",
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )

    result = chat_model._create_chat_result(response)

    assert len(result.generations) == 1
    generation = result.generations[0]
    assert generation.message.content == "This is a test response"
    assert isinstance(generation.message, AIMessage)
    assert generation.generation_info is not None
    assert generation.generation_info["finish_reason"] == "stop"

    assert generation.message.usage_metadata is not None
    assert generation.message.usage_metadata["input_tokens"] == 10
    assert generation.message.usage_metadata["output_tokens"] == 5
    assert generation.message.usage_metadata["total_tokens"] == 15

    assert result.llm_output is not None
    assert result.llm_output["model"] == "gpt-4o"
    assert result.llm_output["token_usage"] is not None
    assert result.llm_output["token_usage"].prompt_tokens == 10


def test_chat_anyllm_bind_tools() -> None:
    """Test that ChatAnyLLM properly binds tools and converts them to OpenAI format."""

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Weather in {location}"

    @tool
    def calculate(expression: str) -> float:
        """Calculate a mathematical expression."""
        return 42.0

    chat_model = ChatAnyLLM(
        model="gpt-4o",
        api_key=None,
        api_base=None,
        model_kwargs={},
    )

    bound_model = chat_model.bind_tools([get_weather, calculate], tool_choice="auto")
    assert hasattr(bound_model, "kwargs")
    assert "tools" in bound_model.kwargs
    assert "tool_choice" in bound_model.kwargs
    assert bound_model.kwargs["tool_choice"] == "auto"

    tools = bound_model.kwargs["tools"]
    assert len(tools) == 2
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"
    assert "location" in tools[0]["function"]["parameters"]["properties"]
    assert tools[1]["function"]["name"] == "calculate"
