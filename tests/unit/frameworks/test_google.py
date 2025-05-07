from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tools import (
    search_web,
    show_final_output,
    visit_webpage,
)


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
        AnyAgent.create(AgentFramework.GOOGLE, AgentConfig(model_id="gpt-4o"))
        mock_agent.assert_called_once_with(
            name="any_agent",
            instruction="",
            model=mock_model(model="gpt-4o"),
            tools=[MockedFunctionTool(search_web), MockedFunctionTool(visit_webpage)],
            sub_agents=[],
            output_key="response",
        )


def test_load_google_multiagent() -> None:
    from google.adk.tools import FunctionTool

    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_agent_tool = MagicMock()
    mock_function_tool = MagicMock()

    class MockedFunctionTool(FunctionTool):
        def __new__(cls, *args: Any, **kwargs: Any) -> "MockedFunctionTool":
            return mock_function_tool

    with (
        patch("any_agent.frameworks.google.LlmAgent", mock_agent),
        patch("any_agent.frameworks.google.DEFAULT_MODEL_TYPE", mock_model),
        patch("any_agent.frameworks.google.AgentTool", mock_agent_tool),
        patch("google.adk.tools.FunctionTool", MockedFunctionTool),
    ):
        AnyAgent.create(
            AgentFramework.GOOGLE,
            AgentConfig(model_id="gpt-4o"),
            managed_agents=[
                AgentConfig(
                    model_id="gpt-4o-mini",
                    name="search-web-agent",
                    tools=[
                        search_web,
                        visit_webpage,
                    ],
                ),
                AgentConfig(
                    model_id="gpt-4o-mini",
                    name="communication-agent",
                    tools=[show_final_output],
                    agent_args={"handoff": True},
                ),
            ],
        )

        mock_agent.assert_any_call(
            model=mock_model(model="gpt-4o-mini"),
            instruction="",
            name="search-web-agent",
            tools=[MockedFunctionTool(search_web), MockedFunctionTool(visit_webpage)],
        )
        mock_agent.assert_any_call(
            model=mock_model(model="gpt-4o-mini"),
            instruction="",
            name="communication-agent",
            tools=[MockedFunctionTool(show_final_output)],
        )
        mock_agent.assert_any_call(
            name="any_agent",
            instruction="",
            model=mock_model(model="gpt-4o"),
            tools=[mock_agent_tool.return_value],
            sub_agents=[mock_agent.return_value],
            output_key="response",
        )


def test_load_google_agent_missing() -> None:
    with patch("any_agent.frameworks.google.adk_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(AgentFramework.GOOGLE, AgentConfig(model_id="gpt-4o"))


def test_run_google_custom_args() -> None:
    from google.adk.agents.run_config import RunConfig
    from google.genai import types

    mock_agent = MagicMock()
    mock_runner = MagicMock()
    mock_runner.session_service.get_session.return_value = AsyncMock()

    run_config = RunConfig(max_llm_calls=10)
    with (
        patch("any_agent.frameworks.google.LlmAgent", mock_agent),
        patch("any_agent.frameworks.google.InMemoryRunner", mock_runner),
        patch("any_agent.frameworks.google.DEFAULT_MODEL_TYPE"),
        patch("google.adk.tools.FunctionTool"),
    ):
        agent = AnyAgent.create(AgentFramework.GOOGLE, AgentConfig(model_id="gpt-4o"))
        agent.run("foo", user_id="1", session_id="2", run_config=run_config)
        mock_runner.return_value.run_async.assert_called_once_with(
            user_id="1",
            session_id="2",
            new_message=types.Content(role="user", parts=[types.Part(text="foo")]),
            run_config=run_config,
        )
