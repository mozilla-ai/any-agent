import os
from unittest.mock import MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tools import (
    ask_user_verification,
    search_web,
    show_final_answer,
    visit_webpage,
)


def test_load_openai_default() -> None:
    mock_agent = MagicMock()
    mock_function_tool = MagicMock()

    with (
        patch("any_agent.frameworks.openai.Agent", mock_agent),
        patch("agents.function_tool", mock_function_tool),
    ):
        AnyAgent.create(AgentFramework.OPENAI, AgentConfig(model_id="gpt-4o"))

        mock_agent.assert_called_once_with(
            name="any_agent",
            model="gpt-4o",
            instructions=None,
            handoffs=[],
            tools=[mock_function_tool(search_web), mock_function_tool(visit_webpage)],
            mcp_servers=[],
        )


def test_openai_with_api_base_and_api_key_var() -> None:
    mock_agent = MagicMock()
    async_openai_mock = MagicMock()
    openai_chat_completions_model = MagicMock()
    with (
        patch("any_agent.frameworks.openai.Agent", mock_agent),
        patch("any_agent.frameworks.openai.AsyncOpenAI", async_openai_mock),
        patch(
            "any_agent.frameworks.openai.OpenAIChatCompletionsModel",
            openai_chat_completions_model,
        ),
        patch.dict(os.environ, {"TEST_API_KEY": "test-key-12345"}),
    ):
        AnyAgent.create(
            AgentFramework.OPENAI,
            AgentConfig(
                model_id="gpt-4o",
                model_args={"base_url": "FOO", "api_key_var": "TEST_API_KEY"},
            ),
        )

        async_openai_mock.assert_called_once_with(
            api_key="test-key-12345",
            base_url="FOO",
        )
        openai_chat_completions_model.assert_called_once()


def test_openai_environment_error() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError, match="MISSING_KEY"):
            AnyAgent.create(
                AgentFramework.OPENAI,
                AgentConfig(
                    model_id="gpt-4o",
                    model_args={"base_url": "FOO", "api_key_var": "MISSING_KEY"},
                ),
            )


def test_load_openai_with_mcp_server() -> None:
    mock_agent = MagicMock()
    mock_function_tool = MagicMock()
    mock_mcp_server = MagicMock()
    mock_mcp_server.server = MagicMock()

    with (
        patch("any_agent.frameworks.openai.Agent", mock_agent),
        patch("agents.function_tool", mock_function_tool),
        patch("any_agent.frameworks.openai.wrap_tools") as mock_wrap_tools,
    ):
        # Setup the mock to return tools and MCP servers
        mock_wrap_tools.return_value = (
            [mock_function_tool(search_web)],  # tools
            [mock_mcp_server],  # mcp_servers
        )

        AnyAgent.create(
            AgentFramework.OPENAI,
            AgentConfig(
                model_id="gpt-4o",
                tools=[
                    "some.mcp.server",
                ],  # The actual import path doesn't matter for the test
            ),
        )

        # Verify Agent was called with the MCP server
        mock_agent.assert_called_once_with(
            name="any_agent",
            model="gpt-4o",
            instructions=None,
            handoffs=[],
            tools=[mock_function_tool(search_web)],
            mcp_servers=[mock_mcp_server.server],
        )


def test_load_openai_multiagent() -> None:
    mock_agent = MagicMock()
    mock_function_tool = MagicMock()

    with (
        patch("any_agent.frameworks.openai.Agent", mock_agent),
        patch("agents.function_tool", mock_function_tool),
    ):
        main_agent = AgentConfig(
            model_id="o3-mini",
        )

        managed_agents = [
            AgentConfig(
                model_id="gpt-4o-mini",
                name="user-verification-agent",
                tools=[ask_user_verification],
            ),
            AgentConfig(
                model_id="gpt-4o",
                name="search-web-agent",
                tools=[
                    search_web,
                    visit_webpage,
                ],
            ),
            AgentConfig(
                model_id="gpt-4o-mini",
                name="communication-agent",
                tools=[show_final_answer],
                handoff=True,
            ),
        ]

        AnyAgent.create(
            AgentFramework.OPENAI,
            main_agent,
            managed_agents=managed_agents,
        )

        mock_agent.assert_any_call(
            model="gpt-4o-mini",
            instructions=None,
            name="user-verification-agent",
            tools=[
                mock_function_tool(ask_user_verification),
            ],
            mcp_servers=[],
        )

        mock_agent.assert_any_call(
            model="gpt-4o",
            instructions=None,
            name="search-web-agent",
            tools=[mock_function_tool(search_web), mock_function_tool(visit_webpage)],
            mcp_servers=[],
        )

        mock_agent.assert_any_call(
            model="gpt-4o-mini",
            instructions=None,
            name="communication-agent",
            tools=[mock_function_tool(show_final_answer)],
            mcp_servers=[],
        )

        mock_agent.assert_any_call(
            model="o3-mini",
            instructions=None,
            name="any_agent",
            handoffs=[mock_agent.return_value],
            tools=[
                mock_agent.return_value.as_tool.return_value,
                mock_agent.return_value.as_tool.return_value,
            ],
            mcp_servers=[],
        )


def test_load_openai_agent_missing() -> None:
    with patch("any_agent.frameworks.openai.agents_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(AgentFramework.OPENAI, AgentConfig(model_id="gpt-4o"))
