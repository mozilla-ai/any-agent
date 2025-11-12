from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from any_llm import AnyLLM, LLMProvider
from pydantic import BaseModel

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.frameworks.tinyagent import TinyAgent, ToolExecutor
from any_agent.testing.helpers import DEFAULT_SMALL_MODEL_ID, LLM_IMPORT_PATHS


class SampleOutput(BaseModel):
    """Test output model for structured output testing."""

    answer: str
    confidence: float


async def sample_tool_function(arg1: int, arg2: str) -> str:
    """Sample tool function for testing argument casting."""
    assert isinstance(arg1, int), "arg1 should be an int"
    assert isinstance(arg2, str), "arg2 should be a str"
    return f"Received int: {arg1}, str: {arg2}"


@pytest.mark.asyncio
async def test_tool_argument_casting_in_agent_flow() -> None:
    """Test that argument casting happens in the main agent flow during tool execution."""
    config = AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID, tools=[sample_tool_function])
    agent: TinyAgent = await AnyAgent.create_async(AgentFramework.TINYAGENT, config)  # type: ignore[assignment]

    def create_mock_tool_response() -> MagicMock:
        """Create a mock LLM response that calls our tool with string arguments."""
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.role = "assistant"

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_function = MagicMock()
        mock_function.name = "sample_tool_function"
        mock_function.arguments = (
            '{"arg1": "42", "arg2": 100}'  # String and int that need casting
        )
        mock_tool_call.function = mock_function
        mock_message.tool_calls = [mock_tool_call]

        mock_message.model_dump.return_value = {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "sample_tool_function",
                        "arguments": '{"arg1": "42", "arg2": 100}',
                    },
                    "type": "function",
                }
            ],
        }
        return MagicMock(choices=[MagicMock(message=mock_message)])

    def create_mock_final_response() -> MagicMock:
        """Create a mock final answer response."""
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.role = "assistant"

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_final"
        mock_function = MagicMock()
        mock_function.name = "final_answer"
        mock_function.arguments = '{"answer": "Task completed successfully"}'
        mock_tool_call.function = mock_function
        mock_message.tool_calls = [mock_tool_call]

        mock_message.model_dump.return_value = {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_final",
                    "function": {
                        "name": "final_answer",
                        "arguments": '{"answer": "Task completed successfully"}',
                    },
                    "type": "function",
                }
            ],
        }
        return MagicMock(choices=[MagicMock(message=mock_message)])

    with patch(LLM_IMPORT_PATHS[AgentFramework.TINYAGENT]) as mock_acompletion:
        # First call returns tool call, second call returns final answer
        mock_acompletion.side_effect = [
            create_mock_tool_response(),
            create_mock_final_response(),
        ]

        result = await agent.run_async("Test the casting")

        assert result.final_output == "Task completed successfully"

        assert mock_acompletion.call_count == 2


@pytest.mark.asyncio
async def test_tool_executor_without_casting() -> None:
    """Test that ToolExecutor no longer does casting - demonstrates the change."""
    agent: TinyAgent = await AnyAgent.create_async(
        AgentFramework.TINYAGENT, AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID)
    )  # type: ignore[assignment]

    agent.clients["sample_tool"] = ToolExecutor(sample_tool_function)

    request_uncast = {
        "name": "sample_tool",
        "arguments": {
            "arg1": "42",  # String instead of int
            "arg2": 100,  # Int instead of str
        },
    }

    result = await agent.clients["sample_tool"].call_tool(request_uncast)
    assert "Error calling tool" in result
    assert "arg1 should be an int" in result

    request_typed = {
        "name": "sample_tool",
        "arguments": {
            "arg1": 42,
            "arg2": "100",
        },
    }

    result = await agent.clients["sample_tool"].call_tool(request_typed)
    assert result == "Received int: 42, str: 100"


def test_run_tinyagent_agent_custom_args() -> None:
    create_mock = MagicMock()
    agent_mock = AsyncMock()
    agent_mock.ainvoke.return_value = MagicMock()
    create_mock.return_value = agent_mock
    output = "The state capital of Pennsylvania is Harrisburg."

    agent = AnyAgent.create(
        AgentFramework.TINYAGENT, AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID)
    )
    with patch(LLM_IMPORT_PATHS[AgentFramework.TINYAGENT]) as mock_acompletion:
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = output
        mock_message.role = "assistant"
        mock_message.tool_calls = []
        mock_response.choices = [MagicMock(message=mock_message)]

        mock_acompletion.return_value = mock_response

        result = agent.run("what's the state capital of Pennsylvania", debug=True)

        assert output == result.final_output


def test_output_type_completion_params_isolation() -> None:
    """Test that completion_params are not polluted between calls when using output_type."""
    config = AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID, output_type=SampleOutput)
    agent: TinyAgent = AnyAgent.create(AgentFramework.TINYAGENT, config)  # type: ignore[assignment]
    original_completion_params = agent.completion_params.copy()

    def create_mock_response(content: str, is_structured: bool = False) -> MagicMock:
        """Helper to create mock responses."""
        mock_message = MagicMock()
        mock_message.content = content
        mock_message.role = "assistant"
        mock_message.tool_calls = []
        mock_message.model_dump.return_value = {
            "content": content,
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
            "annotations": [],
        }
        if is_structured:
            mock_message.__getitem__.return_value = content
        return MagicMock(choices=[MagicMock(message=mock_message)])

    with patch(LLM_IMPORT_PATHS[AgentFramework.TINYAGENT]) as mock_acompletion:
        # Mock responses: 2 calls per run (regular + structured output)
        mock_acompletion.side_effect = [
            create_mock_response("First response"),  # First run, regular call
            create_mock_response(
                '{"answer": "First response", "confidence": 0.9}', True
            ),  # First run, structured
        ]

        # First call - should trigger structured output handling
        agent.run("First question")

        # Verify completion_params weren't modified
        assert agent.completion_params == original_completion_params


def test_structured_output_without_tools() -> None:
    """Test that structured output works correctly when no tools are present and tool_choice is not set."""
    config = AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID, output_type=SampleOutput)
    agent: TinyAgent = AnyAgent.create(AgentFramework.TINYAGENT, config)  # type: ignore[assignment]

    def create_mock_response(content: str, is_structured: bool = False) -> MagicMock:
        """Helper to create mock responses."""
        mock_message = MagicMock()
        mock_message.content = content
        mock_message.role = "assistant"
        mock_message.tool_calls = []
        mock_message.model_dump.return_value = {
            "content": content,
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
            "annotations": [],
        }
        if is_structured:
            mock_message.__getitem__.return_value = content
        return MagicMock(choices=[MagicMock(message=mock_message)])

    with patch(LLM_IMPORT_PATHS[AgentFramework.TINYAGENT]) as mock_acompletion:
        # Mock responses: 2 calls per run (regular + structured output)
        mock_acompletion.side_effect = [
            create_mock_response("Initial response"),  # First call - regular response
            create_mock_response(
                '{"answer": "Structured answer", "confidence": 0.95}', True
            ),  # Second call - structured output
        ]

        # Run the agent
        agent.run("Test question")

        # Verify that acompletion was called twice. Once for the regular output and once for the structured output.
        assert mock_acompletion.call_count == 2

        # Get the call arguments for the second call (structured output)
        second_call_args = mock_acompletion.call_args_list[1][1]

        # tool choice should not be set when no tools are present
        assert "tool_choice" not in second_call_args

        # Verify that response_format is set for structured output
        assert "response_format" in second_call_args
        assert second_call_args["response_format"] == SampleOutput


@pytest.mark.parametrize(
    ("model_id", "expected_uses_openai"),
    [
        ("gateway:openai:gpt-4.1-mini", True),
        ("gateway:anthropic:claude-3", False),
        ("openai:gpt-4", True),
        ("anthropic:claude-3", False),
    ],
)
def test_uses_openai_handles_gateway_provider(
    model_id: str, expected_uses_openai: bool
) -> None:
    config = AgentConfig(model_id=model_id)
    agent: TinyAgent = AnyAgent.create(AgentFramework.TINYAGENT, config)  # type: ignore[assignment]

    assert agent.uses_openai is expected_uses_openai
