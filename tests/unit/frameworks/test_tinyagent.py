from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.frameworks.tinyagent import TinyAgent, ToolExecutor


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
    assert result["content"][0]["text"] == "Received int: 42, str: 100"


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
