from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent


def test_load_google_default() -> None:
    from google.adk.tools import FunctionTool

    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_function_tool = MagicMock()

    class MockedFunctionTool(FunctionTool):
        def __new__(cls, *args: Any, **kwargs: Any) -> MagicMock:
            return mock_function_tool

    with (
        patch("any_agent.frameworks.google.LlmAgent", mock_agent),
        patch("any_agent.frameworks.google.DEFAULT_MODEL_TYPE", mock_model),
        patch("google.adk.tools.FunctionTool", MockedFunctionTool),
    ):
        AnyAgent.create(
            AgentFramework.GOOGLE, AgentConfig(model_id="mistral/mistral-small-latest")
        )
        mock_agent.assert_called_once_with(
            name="any_agent",
            instruction="",
            model=mock_model(model="mistral/mistral-small-latest"),
            tools=[],
            output_key="response",
        )


def test_load_google_agent_missing() -> None:
    with patch("any_agent.frameworks.google.adk_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(
                AgentFramework.GOOGLE,
                AgentConfig(model_id="mistral/mistral-small-latest"),
            )


def test_run_google_custom_args() -> None:
    from google.adk.agents.run_config import RunConfig
    from google.genai import types

    mock_agent = MagicMock()
    mock_runner = MagicMock()
    mock_runner.get_tools = AsyncMock()
    mock_session = MagicMock()
    mock_runner.return_value.session_service.create_session = AsyncMock()
    mock_runner.return_value.session_service.get_session = AsyncMock()

    # More explicit mock setup
    mock_state = MagicMock()
    mock_state.get.return_value = "mock response"
    mock_session.state = mock_state
    mock_runner.return_value.session_service.get_session.return_value = mock_session

    run_config = RunConfig(max_llm_calls=10)
    with (
        patch("any_agent.frameworks.google.LlmAgent", mock_agent),
        patch("any_agent.frameworks.google.InMemoryRunner", mock_runner),
        patch("any_agent.frameworks.google.DEFAULT_MODEL_TYPE"),
        patch("google.adk.tools.FunctionTool"),
    ):
        agent = AnyAgent.create(
            AgentFramework.GOOGLE, AgentConfig(model_id="mistral/mistral-small-latest")
        )
        result = agent.run("foo", user_id="1", session_id="2", run_config=run_config)

        # Verify the result is as expected
        assert isinstance(result.final_output, str)
        assert result.final_output == "mock response"

        mock_runner.return_value.run_async.assert_called_once_with(
            user_id="1",
            session_id="2",
            new_message=types.Content(role="user", parts=[types.Part(text="foo")]),
            run_config=run_config,
        )
