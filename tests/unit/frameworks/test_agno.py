from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent


def test_load_agno_default() -> None:
    mock_agent = MagicMock()
    mock_model = MagicMock()

    with (
        patch("any_agent.frameworks.agno.Agent", mock_agent),
        patch("any_agent.frameworks.agno.DEFAULT_MODEL_TYPE", mock_model),
    ):
        AnyAgent.create(
            AgentFramework.AGNO, AgentConfig(model_id="mistral/mistral-small-latest")
        )
        mock_agent.assert_called_once_with(
            name="any_agent",
            instructions=None,
            model=mock_model(model="mistral/mistral-small-latest"),
            tools=[],
        )


def test_load_agno_agent_missing() -> None:
    with patch("any_agent.frameworks.agno.agno_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(
                AgentFramework.AGNO,
                AgentConfig(model_id="mistral/mistral-small-latest"),
            )


def test_run_agno_custom_args() -> None:
    mock_agent = MagicMock()
    # Create a mock response object with the required content attribute
    mock_response = MagicMock()
    mock_response.content = "mock response"

    # Set up the AsyncMock to return the mock response
    mock_agent_instance = AsyncMock()
    mock_agent_instance.arun.return_value = mock_response
    mock_agent.return_value = mock_agent_instance

    mock_model = MagicMock()

    with (
        patch("any_agent.frameworks.agno.Agent", mock_agent),
        patch("any_agent.frameworks.agno.DEFAULT_MODEL_TYPE", mock_model),
    ):
        agent = AnyAgent.create(
            AgentFramework.AGNO, AgentConfig(model_id="mistral/mistral-small-latest")
        )
        result = agent.run("foo", retries=2)

        # Verify the result is as expected
        assert isinstance(result.final_output, str)
        assert result.final_output == "mock response"

        # Verify the agent was called with the right parameters
        mock_agent_instance.arun.assert_called_once_with("foo", retries=2)
