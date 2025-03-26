from unittest.mock import patch, MagicMock

from any_agent.runners.openai import run_openai_agent


def test_run_openai_default():
    runner_mock = MagicMock()
    with patch("any_agent.runners.openai.Runner", runner_mock):
        agent_mock = MagicMock()
        run_openai_agent(agent_mock, "Test Query")
        runner_mock.run_sync.assert_called_with(agent_mock, "Test Query")
