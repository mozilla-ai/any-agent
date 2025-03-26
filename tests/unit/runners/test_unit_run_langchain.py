from unittest.mock import patch, MagicMock

from any_agent.runners.langchain import run_langchain_agent


def test_run_langchain_default():
    executor_mock = MagicMock()
    with patch("any_agent.runners.langchain.AgentExecutor", executor_mock):
        agent_mock = MagicMock()
        run_langchain_agent(agent_mock, "Test Query")
        executor_mock.assert_called_with(agent=agent_mock, tools=agent_mock.tools)
        executor_mock.return_value.invoke.assert_called_with("Test Query")
