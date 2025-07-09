from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.frameworks.tinyagent import TinyAgent, ToolExecutor
from any_agent.testing.helpers import LITELLM_IMPORT_PATHS


class SampleOutput(BaseModel):
    """Test output model for structured output testing."""

    answer: str
    confidence: float


async def sample_tool_function(arg1: int, arg2: str) -> str:
    assert isinstance(arg1, int), "arg1 should be an int"
    assert isinstance(arg2, str), "arg2 should be a str"
    return f"Received int: {arg1}, str: {arg2}"


@pytest.mark.asyncio
async def test_tool_argument_casting() -> None:
    agent: TinyAgent = await AnyAgent.create_async(
        AgentFramework.TINYAGENT, AgentConfig(model_id="mistral/mistral-small-latest")
    )  # type: ignore[assignment]

    # Register the sample tool function
    agent.clients["sample_tool"] = ToolExecutor(sample_tool_function)

    request = {
        "name": "sample_tool",
        "arguments": {
            "arg1": "42",  # This should be cast to int
            "arg2": 100,  # This should be cast to str
        },
    }

    # Call the tool and get the result
    result = await agent.clients["sample_tool"].call_tool(request)
    # Check the result
    assert result == "Received int: 42, str: 100"


def test_run_tinyagent_agent_custom_args() -> None:
    create_mock = MagicMock()
    agent_mock = AsyncMock()
    agent_mock.ainvoke.return_value = MagicMock()
    create_mock.return_value = agent_mock
    output = "The state capital of Pennsylvania is Harrisburg."

    agent = AnyAgent.create(
        AgentFramework.TINYAGENT, AgentConfig(model_id="mistral/mistral-small-latest")
    )
    with patch(
        "any_agent.frameworks.tinyagent.litellm.acompletion"
    ) as mock_acompletion:
        # Create a mock response object that properly mocks the LiteLLM response structure
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = output
        mock_message.tool_calls = []  # No tool calls in this response
        mock_message.model_dump.return_value = {
            "content": output,
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
            "annotations": [],
        }

        mock_response.choices = [MagicMock(message=mock_message)]

        # Make the acompletion function return this response
        mock_acompletion.return_value = mock_response

        # Call run which will eventually call _process_single_turn_with_tools
        result = agent.run("what's the state capital of Pennsylvania", debug=True)

        # Assert that the result contains the expected content
        assert output == result.final_output


def test_output_type_completion_params_isolation() -> None:
    """Test that completion_params are not polluted between calls when using output_type."""
    config = AgentConfig(model_id="gpt-4o", output_type=SampleOutput)
    agent: TinyAgent = AnyAgent.create(AgentFramework.TINYAGENT, config)  # type: ignore[assignment]
    original_completion_params = agent.completion_params.copy()

    def create_mock_response(content: str, is_structured: bool = False) -> MagicMock:
        """Helper to create mock responses."""
        mock_message = MagicMock()
        mock_message.content = content
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

    with patch(
        "any_agent.frameworks.tinyagent.litellm.acompletion"
    ) as mock_acompletion:
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
    config = AgentConfig(model_id="gpt-4.1-mini", output_type=SampleOutput)
    agent: TinyAgent = AnyAgent.create(AgentFramework.TINYAGENT, config)  # type: ignore[assignment]

    def create_mock_response(content: str, is_structured: bool = False) -> MagicMock:
        """Helper to create mock responses."""
        mock_message = MagicMock()
        mock_message.content = content
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

    with patch(LITELLM_IMPORT_PATHS[AgentFramework.TINYAGENT]) as mock_acompletion:
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

        # tool choice should not be set to none when no tools are present
        assert second_call_args["tool_choice"] == "auto"

        # Verify that response_format is set for structured output
        assert "response_format" in second_call_args
        assert second_call_args["response_format"] == SampleOutput
