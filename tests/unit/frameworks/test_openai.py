from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import MCPStdio
from any_agent.tools import (
    search_web,
)


def test_load_openai_default() -> None:
    mock_agent = MagicMock()
    mock_function_tool = MagicMock()
    mock_litellm_model = MagicMock()

    with (
        patch("any_agent.frameworks.openai.Agent", mock_agent),
        patch("agents.function_tool", mock_function_tool),
        patch("any_agent.frameworks.openai.DEFAULT_MODEL_TYPE", mock_litellm_model),
    ):
        AnyAgent.create(
            AgentFramework.OPENAI, AgentConfig(model_id="mistral/mistral-small-latest")
        )

        mock_litellm_model.assert_called_once_with(
            model="mistral/mistral-small-latest",
            base_url=None,
            api_key=None,
        )
        mock_agent.assert_called_once_with(
            name="any_agent",
            model=mock_litellm_model.return_value,
            instructions=None,
            tools=[],
            mcp_servers=[],
        )


def test_openai_with_api_base() -> None:
    mock_agent = MagicMock()
    litllm_model_mock = MagicMock()
    with (
        patch("any_agent.frameworks.openai.Agent", mock_agent),
        patch(
            "any_agent.frameworks.openai.DEFAULT_MODEL_TYPE",
            litllm_model_mock,
        ),
    ):
        AnyAgent.create(
            AgentFramework.OPENAI,
            AgentConfig(
                model_id="mistral/mistral-small-latest", model_args={}, api_base="FOO"
            ),
        )
        litllm_model_mock.assert_called_once_with(
            model="mistral/mistral-small-latest",
            base_url="FOO",
            api_key=None,
        )


def test_openai_with_api_key() -> None:
    mock_agent = MagicMock()
    litellm_model_mock = MagicMock()
    with (
        patch("any_agent.frameworks.openai.Agent", mock_agent),
        patch(
            "any_agent.frameworks.openai.DEFAULT_MODEL_TYPE",
            litellm_model_mock,
        ),
    ):
        AnyAgent.create(
            AgentFramework.OPENAI,
            AgentConfig(
                model_id="mistral/mistral-small-latest", model_args={}, api_key="FOO"
            ),
        )
        litellm_model_mock.assert_called_once_with(
            model="mistral/mistral-small-latest",
            base_url=None,
            api_key="FOO",
        )


def test_load_openai_with_mcp_server() -> None:
    mock_agent = MagicMock()
    mock_function_tool = MagicMock()
    mock_mcp_server = MagicMock()
    mock_mcp_server.server = MagicMock()
    mock_litellm_model = MagicMock()
    mock_wrap_tools = MagicMock()

    with (
        patch("any_agent.frameworks.openai.Agent", mock_agent),
        patch("agents.function_tool", mock_function_tool),
        patch("any_agent.frameworks.openai.DEFAULT_MODEL_TYPE", mock_litellm_model),
        patch.object(AnyAgent, "_load_tools", mock_wrap_tools),
    ):

        async def side_effect(self):  # type: ignore[no-untyped-def]
            return (
                [mock_function_tool(search_web)],  # tools
                [mock_mcp_server],  # mcp_servers
            )

        mock_wrap_tools.side_effect = side_effect

        AnyAgent.create(
            AgentFramework.OPENAI,
            AgentConfig(
                model_id="mistral/mistral-small-latest",
                tools=[
                    MCPStdio(
                        command="docker",
                        args=["run", "-i", "--rm", "mcp/fetch"],
                        tools=["fetch"],
                    ),
                ],
            ),
        )

        # Verify Agent was called with the MCP server
        mock_agent.assert_called_once_with(
            name="any_agent",
            model=mock_litellm_model.return_value,
            instructions=None,
            tools=[mock_function_tool(search_web)],
            mcp_servers=[mock_mcp_server.server],
        )


def test_load_openai_agent_missing() -> None:
    with patch("any_agent.frameworks.openai.agents_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(
                AgentFramework.OPENAI,
                AgentConfig(model_id="mistral/mistral-small-latest"),
            )


def test_run_openai_with_custom_args() -> None:
    mock_agent = MagicMock()
    mock_runner = AsyncMock()

    with (
        patch("any_agent.frameworks.openai.Runner", mock_runner),
        patch("any_agent.frameworks.openai.Agent", mock_agent),
        patch("agents.function_tool"),
        patch("any_agent.frameworks.openai.DEFAULT_MODEL_TYPE"),
    ):
        agent = AnyAgent.create(
            AgentFramework.OPENAI, AgentConfig(model_id="mistral/mistral-small-latest")
        )
        agent.run("foo", max_turns=30)
        mock_runner.run.assert_called_once_with(
            mock_agent.return_value, "foo", max_turns=30
        )
