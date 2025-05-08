from unittest.mock import AsyncMock, MagicMock, patch

from any_agent import AgentConfig, AgentFramework, AnyAgent


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
