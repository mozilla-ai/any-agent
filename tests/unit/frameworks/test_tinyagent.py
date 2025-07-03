from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.frameworks.tinyagent import TinyAgent, ToolExecutor


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
        AgentFramework.TINYAGENT, AgentConfig(model_id="gpt-4o")
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

    agent = AnyAgent.create(AgentFramework.TINYAGENT, AgentConfig(model_id="gpt-4o"))
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
    # Create agent with output_type
    config = AgentConfig(model_id="gpt-4o", output_type=SampleOutput)
    agent = AnyAgent.create(AgentFramework.TINYAGENT, config)
    original_completion_params = agent.completion_params.copy()

    with patch(
        "any_agent.frameworks.tinyagent.litellm.acompletion"
    ) as mock_acompletion:
        # Mock responses for the first run (2 calls: regular + structured output)
        mock_response1 = MagicMock()
        mock_message1 = MagicMock()
        mock_message1.content = "First response"
        mock_message1.tool_calls = []
        mock_message1.model_dump.return_value = {
            "content": "First response",
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
            "annotations": [],
        }
        mock_response1.choices = [MagicMock(message=mock_message1)]

        # Mock response for the structured output call
        mock_response_structured = MagicMock()
        mock_message_structured = MagicMock()
        mock_message_structured.content = (
            '{"answer": "First response", "confidence": 0.9}'
        )
        mock_message_structured.tool_calls = []
        mock_message_structured.model_dump.return_value = {
            "content": '{"answer": "First response", "confidence": 0.9}',
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
            "annotations": [],
        }
        # Configure the mock to return the content string when accessed as dict
        mock_message_structured.__getitem__.return_value = (
            '{"answer": "First response", "confidence": 0.9}'
        )
        mock_response_structured.choices = [MagicMock(message=mock_message_structured)]

        # Mock responses for the second run (2 calls: regular + structured output)
        mock_response2 = MagicMock()
        mock_message2 = MagicMock()
        mock_message2.content = "Second response"
        mock_message2.tool_calls = []
        mock_message2.model_dump.return_value = {
            "content": "Second response",
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
            "annotations": [],
        }
        mock_response2.choices = [MagicMock(message=mock_message2)]

        mock_response_structured2 = MagicMock()
        mock_message_structured2 = MagicMock()
        mock_message_structured2.content = (
            '{"answer": "Second response", "confidence": 0.95}'
        )
        mock_message_structured2.tool_calls = []
        mock_message_structured2.model_dump.return_value = {
            "content": '{"answer": "Second response", "confidence": 0.95}',
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
            "annotations": [],
        }
        # Configure the mock to return the content string when accessed as dict
        mock_message_structured2.__getitem__.return_value = (
            '{"answer": "Second response", "confidence": 0.95}'
        )
        mock_response_structured2.choices = [
            MagicMock(message=mock_message_structured2)
        ]

        # Make acompletion return responses in order: first run (2 calls), then second run (2 calls)
        mock_acompletion.side_effect = [
            mock_response1,  # First run, first call
            mock_response_structured,  # First run, structured output call
            mock_response2,  # Second run, first call
            mock_response_structured2,  # Second run, structured output call
        ]

        # First call - should trigger structured output handling
        agent.run("First question")

        assert agent.completion_params == original_completion_params
