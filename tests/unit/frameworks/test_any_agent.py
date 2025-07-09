# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
from typing import Any
from unittest.mock import patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.testing.helpers import LITELLM_IMPORT_PATHS

TEST_TEMPERATURE = 0.54321
TEST_PENALTY = 0.5
TEST_QUERY = "what's the state capital of Pennsylvania"
EXPECTED_OUTPUT = "The state capital of Pennsylvania is Harrisburg."


def create_agent_with_model_args(framework: AgentFramework) -> AnyAgent:
    """Helper function to create an agent with test model arguments"""
    return AnyAgent.create(
        framework,
        AgentConfig(
            model_id="mistral/mistral-small-latest",
            model_args={
                "temperature": TEST_TEMPERATURE,
                "frequency_penalty": TEST_PENALTY,
            },
        ),
    )


def test_create_any_with_framework(agent_framework: AgentFramework) -> None:
    agent = AnyAgent.create(
        agent_framework, AgentConfig(model_id="mistral/mistral-small-latest")
    )
    assert agent


def test_create_any_with_valid_string(agent_framework: AgentFramework) -> None:
    agent = AnyAgent.create(
        agent_framework.name, AgentConfig(model_id="mistral/mistral-small-latest")
    )
    assert agent


def test_create_any_with_invalid_string() -> None:
    with pytest.raises(ValueError, match="Unsupported agent framework"):
        AnyAgent.create(
            "non-existing", AgentConfig(model_id="mistral/mistral-small-latest")
        )


def test_model_args(
    agent_framework: AgentFramework,
    mock_litellm_response: Any,
) -> None:
    if agent_framework == AgentFramework.LLAMA_INDEX:
        pytest.skip("LlamaIndex agent uses a litellm streaming syntax")

    agent = create_agent_with_model_args(agent_framework)

    # Patch the appropriate litellm import path for this framework
    import_path = LITELLM_IMPORT_PATHS[agent_framework]
    with patch(import_path, return_value=mock_litellm_response) as mock_litellm:
        # Run the agent
        result = agent.run(TEST_QUERY)

        # Verify results
        assert EXPECTED_OUTPUT == result.final_output
        assert mock_litellm.call_args.kwargs["temperature"] == TEST_TEMPERATURE
        assert mock_litellm.call_args.kwargs["frequency_penalty"] == TEST_PENALTY
        assert mock_litellm.call_count > 0


def test_model_args_streaming(
    agent_framework: AgentFramework, mock_litellm_streaming: Any
) -> None:
    if agent_framework != AgentFramework.LLAMA_INDEX:
        pytest.skip("This test is only for LlamaIndex framework")

    agent = create_agent_with_model_args(agent_framework)

    # Patch the appropriate litellm import path for LlamaIndex
    import_path = LITELLM_IMPORT_PATHS[agent_framework]
    with patch(import_path, side_effect=mock_litellm_streaming) as mock_litellm:
        # Run the agent
        result = agent.run(TEST_QUERY)

        # Verify results
        assert result.final_output
        assert "Harrisburg" in result.final_output
        assert mock_litellm.call_args.kwargs["stream"] is True
        assert mock_litellm.call_args.kwargs["temperature"] == TEST_TEMPERATURE
        assert mock_litellm.call_args.kwargs["frequency_penalty"] == TEST_PENALTY
        assert mock_litellm.call_count > 0
