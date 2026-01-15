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


@pytest.mark.asyncio
async def test_tool_result_appended_when_tool_not_found() -> None:
    """Test that tool_result message is appended when a tool is not found.

    This test verifies that when the LLM calls a tool that doesn't exist,
    the error message is properly appended to the messages list. Without this,
    Anthropic API returns 400 errors about tool_use without tool_result.
    """
    nonexistent_tool_name = "nonexistent_tool"
    nonexistent_tool_call_id = "call_nonexistent"
    nonexistent_tool_args = '{"query": "test"}'
    final_tool_call_id = "call_final"
    final_answer_text = "Done"
    final_tool_args = f'{{"answer": "{final_answer_text}"}}'

    config = AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID, tools=[sample_tool_function])
    agent: TinyAgent = await AnyAgent.create_async(AgentFramework.TINYAGENT, config)  # type: ignore[assignment]

    def create_mock_nonexistent_tool_response() -> MagicMock:
        """Mock a tool call response for a non-existent tool.

        The LLM expects the response to contain a tool_result, even if that
        result is an error. A response with it missing causes the Anthropic
        API to return 400 errors about tool_use without tool_result.
        """
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.role = "assistant"

        mock_tool_call = MagicMock()
        mock_tool_call.id = nonexistent_tool_call_id
        mock_function = MagicMock()
        mock_function.name = nonexistent_tool_name
        mock_function.arguments = nonexistent_tool_args
        mock_tool_call.function = mock_function
        mock_message.tool_calls = [mock_tool_call]

        mock_message.model_dump.return_value = {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": nonexistent_tool_call_id,
                    "function": {
                        "name": nonexistent_tool_name,
                        "arguments": nonexistent_tool_args,
                    },
                    "type": "function",
                }
            ],
        }
        return MagicMock(choices=[MagicMock(message=mock_message)])

    def create_mock_final_response() -> MagicMock:
        """Mock a final_answer tool call to end the agent loop.

        This allows the test to complete successfully after the nonexistent
        tool error is handled.
        """
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.role = "assistant"

        mock_tool_call = MagicMock()
        mock_tool_call.id = final_tool_call_id
        mock_function = MagicMock()
        mock_function.name = "final_answer"
        mock_function.arguments = final_tool_args
        mock_tool_call.function = mock_function
        mock_message.tool_calls = [mock_tool_call]

        mock_message.model_dump.return_value = {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": final_tool_call_id,
                    "function": {
                        "name": "final_answer",
                        "arguments": final_tool_args,
                    },
                    "type": "function",
                }
            ],
        }
        return MagicMock(choices=[MagicMock(message=mock_message)])

    with patch(LLM_IMPORT_PATHS[AgentFramework.TINYAGENT]) as mock_acompletion:
        mock_acompletion.side_effect = [
            create_mock_nonexistent_tool_response(),
            create_mock_final_response(),
        ]

        result = await agent.run_async("Call a tool")

        assert result.final_output == final_answer_text
        assert mock_acompletion.call_count == 2

        # Verify the second call includes the tool_result for the nonexistent tool.
        second_call_messages = mock_acompletion.call_args_list[1][1]["messages"]

        # Find the assistant message containing the tool_use for the nonexistent tool.
        assistant_msg_index = None
        for i, msg in enumerate(second_call_messages):
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls and tool_calls[0].get("id") == nonexistent_tool_call_id:
                    assistant_msg_index = i
                    break

        assert assistant_msg_index is not None

        # Verify tool_result immediately follows the assistant message.
        # Anthropic requires tool_result blocks immediately after tool_use.
        tool_result_msg = second_call_messages[assistant_msg_index + 1]
        assert tool_result_msg.get("role") == "tool"
        assert tool_result_msg.get("tool_call_id") == nonexistent_tool_call_id
        assert tool_result_msg.get("name") == nonexistent_tool_name
        assert (
            f"No tool found with name: {nonexistent_tool_name}"
            in tool_result_msg["content"]
        )
