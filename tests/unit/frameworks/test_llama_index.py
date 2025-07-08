from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent


def test_load_llama_index_agent_default() -> None:
    model_mock = MagicMock()
    create_mock = MagicMock()
    agent_mock = MagicMock()
    create_mock.return_value = agent_mock
    tool_mock = MagicMock()
    from llama_index.core.tools import FunctionTool

    with (
        patch("any_agent.frameworks.llama_index.DEFAULT_AGENT_TYPE", create_mock),
        patch("any_agent.frameworks.llama_index.DEFAULT_MODEL_TYPE", model_mock),
        patch.object(FunctionTool, "from_defaults", tool_mock),
    ):
        AnyAgent.create(
            AgentFramework.LLAMA_INDEX,
            AgentConfig(
                model_id="gemini/gemini-2.0-flash",
                instructions="You are a helpful assistant",
            ),
        )

        model_mock.assert_called_once_with(
            model="gemini/gemini-2.0-flash",
            api_key=None,
            api_base=None,
            additional_kwargs={
                "stream_options": {"include_usage": True},
            },
        )
        create_mock.assert_called_once_with(
            name="any_agent",
            llm=model_mock.return_value,
            system_prompt="You are a helpful assistant",
            description="The main agent",
            tools=[],
        )


def test_load_llama_index_agent_missing() -> None:
    with patch("any_agent.frameworks.llama_index.llama_index_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(
                AgentFramework.LLAMA_INDEX,
                AgentConfig(model_id="mistral/mistral-small-latest"),
            )


def test_run_llama_index_agent_custom_args() -> None:
    create_mock = MagicMock()
    agent_mock = AsyncMock()
    create_mock.return_value = agent_mock
    from llama_index.core.tools import FunctionTool

    with (
        patch("any_agent.frameworks.llama_index.DEFAULT_AGENT_TYPE", create_mock),
        patch("any_agent.frameworks.llama_index.DEFAULT_MODEL_TYPE"),
        patch.object(FunctionTool, "from_defaults"),
    ):
        agent = AnyAgent.create(
            AgentFramework.LLAMA_INDEX,
            AgentConfig(
                model_id="gemini/gemini-2.0-flash",
                instructions="You are a helpful assistant",
            ),
        )
        agent.run("foo", timeout=10)
        agent_mock.run.assert_called_once_with("foo", timeout=10)
