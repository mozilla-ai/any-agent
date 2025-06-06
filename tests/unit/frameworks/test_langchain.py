from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent


def test_load_langchain_agent_default() -> None:
    model_mock = MagicMock()
    create_mock = MagicMock()
    agent_mock = MagicMock()
    create_mock.return_value = agent_mock
    tool_mock = MagicMock()

    with (
        patch("any_agent.frameworks.langchain.DEFAULT_AGENT_TYPE", create_mock),
        patch("any_agent.frameworks.langchain.DEFAULT_MODEL_TYPE", model_mock),
        patch("langchain_core.tools.tool", tool_mock),
    ):
        AnyAgent.create(AgentFramework.LANGCHAIN, AgentConfig(model_id="gpt-4o"))

        model_mock.assert_called_once_with(
            model="gpt-4o", api_base=None, api_key=None, model_kwargs={}
        )
        create_mock.assert_called_once_with(
            name="any_agent",
            model=model_mock.return_value,
            tools=[],
            prompt=None,
        )


def test_load_langchain_agent_missing() -> None:
    with patch("any_agent.frameworks.langchain.langchain_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(AgentFramework.LANGCHAIN, AgentConfig(model_id="gpt-4o"))


def test_run_langchain_agent_custom_args() -> None:
    create_mock = MagicMock()
    agent_mock = AsyncMock()
    agent_mock.ainvoke.return_value = MagicMock()
    create_mock.return_value = agent_mock

    with (
        patch("any_agent.frameworks.langchain.DEFAULT_AGENT_TYPE", create_mock),
        patch("any_agent.frameworks.langchain.DEFAULT_MODEL_TYPE"),
        patch("langchain_core.tools.tool"),
    ):
        agent = AnyAgent.create(
            AgentFramework.LANGCHAIN, AgentConfig(model_id="gpt-4o")
        )
        agent.run("foo", debug=True)
        agent_mock.ainvoke.assert_called_once_with(
            {"messages": [("user", "foo")]}, debug=True, config={"callbacks": [ANY]}
        )
