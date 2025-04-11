from unittest.mock import patch, MagicMock

import pytest

from any_agent import AgentFramework, AgentConfig, AnyAgent
from any_agent.tools import (
    search_web,
    show_final_answer,
    visit_webpage,
)


def test_load_google_default():
    from google.adk.tools import FunctionTool

    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_function_tool = MagicMock()

    class MockedFunctionTool(FunctionTool):
        def __new__(cls, *args, **kwargs):
            return mock_function_tool

    with (
        patch("any_agent.frameworks.google.Agent", mock_agent),
        patch("any_agent.frameworks.google.LiteLlm", mock_model),
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


def test_load_google_multiagent():
    from google.adk.tools import FunctionTool

    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_agent_tool = MagicMock()
    mock_function_tool = MagicMock()

    class MockedFunctionTool(FunctionTool):
        def __new__(cls, *args, **kwargs):
            return mock_function_tool

    with (
        patch("any_agent.frameworks.google.Agent", mock_agent),
        patch("any_agent.frameworks.google.LiteLlm", mock_model),
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
                        "any_agent.tools.search_web",
                        "any_agent.tools.visit_webpage",
                    ],
                ),
                AgentConfig(
                    model_id="gpt-4o-mini",
                    name="communication-agent",
                    tools=["any_agent.tools.show_final_answer"],
                    handoff=True,
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
            tools=[MockedFunctionTool(show_final_answer)],
        )
        mock_agent.assert_any_call(
            name="any_agent",
            instruction="",
            model=mock_model(model="gpt-4o"),
            tools=[mock_agent_tool.return_value],
            sub_agents=[mock_agent.return_value],
            output_key="response",
        )


def test_load_google_agent_missing():
    with patch("any_agent.frameworks.google.adk_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(AgentFramework.GOOGLE, AgentConfig(model_id="gpt-4o"))
