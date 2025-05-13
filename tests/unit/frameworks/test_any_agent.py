# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponse

from any_agent import AgentConfig, AgentFramework, AnyAgent

TEST_TEMPERATURE = 0.54321
TEST_PENALTY = 0.5
TEST_QUERY = "what's the state capital of Pennsylvania"
EXPECTED_OUTPUT = "The state capital of Pennsylvania is Harrisburg."

# Framework-specific LiteLLM import paths
LITELLM_IMPORT_PATHS = {
    AgentFramework.GOOGLE: "google.adk.models.lite_llm.acompletion",
    AgentFramework.LANGCHAIN: "litellm.acompletion",
    AgentFramework.TINYAGENT: "litellm.acompletion",
    AgentFramework.AGNO: "litellm.acompletion",
    AgentFramework.OPENAI: "litellm.acompletion",
    AgentFramework.SMOLAGENTS: "litellm.completion",
    AgentFramework.LLAMA_INDEX: "litellm.acompletion",
}


# Fixtures
@pytest.fixture
def mock_litellm_response() -> ModelResponse:
    """Fixture to create a standard mock LiteLLM response"""
    return ModelResponse.model_validate_json(
        '{"id":"chatcmpl-BWnfbHWPsQp05roQ06LAD1mZ9tOjT","created":1747157127,"model":"gpt-4o-2024-08-06","object":"chat.completion","system_fingerprint":"fp_f5bdcc3276","choices":[{"finish_reason":"stop","index":0,"message":{"content":"The state capital of Pennsylvania is Harrisburg.","role":"assistant","tool_calls":null,"function_call":null,"annotations":[]}}],"usage":{"completion_tokens":11,"prompt_tokens":138,"total_tokens":149,"completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,"reasoning_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens_details":{"audio_tokens":0,"cached_tokens":0}},"service_tier":"default"}'
    )


def create_agent_with_model_args(framework: AgentFramework) -> AnyAgent:
    """Helper function to create an agent with test model arguments"""
    return AnyAgent.create(
        framework,
        AgentConfig(
            model_id="gpt-4o",
            model_args={
                "temperature": TEST_TEMPERATURE,
                "frequency_penalty": TEST_PENALTY,
            },
        ),
    )


async def mock_streaming_response() -> AsyncGenerator[dict[str, Any], None]:
    """
    Mock the streaming response from litellm's acompletion function.
    Accepts all arguments that would be passed to the real acompletion function.
    """
    # First chunk with role
    yield {
        "choices": [
            {
                "delta": {"role": "assistant", "content": "The state "},
                "index": 0,
                "finish_reason": None,
            }
        ]
    }

    # Middle chunks with content
    yield {
        "choices": [
            {"delta": {"content": "capital of "}, "index": 0, "finish_reason": None}
        ]
    }

    yield {
        "choices": [
            {
                "delta": {"content": "Pennsylvania is "},
                "index": 0,
                "finish_reason": None,
            }
        ]
    }

    # Final chunk with finish reason
    yield {
        "choices": [
            {"delta": {"content": "Harrisburg."}, "index": 0, "finish_reason": "stop"}
        ]
    }


# Tests for agent creation
class TestAgentCreation:
    def test_create_any_with_framework(self, agent_framework: AgentFramework) -> None:
        agent = AnyAgent.create(agent_framework, AgentConfig(model_id="gpt-4o"))
        assert agent

    def test_create_any_with_valid_string(
        self, agent_framework: AgentFramework
    ) -> None:
        agent = AnyAgent.create(agent_framework.name, AgentConfig(model_id="gpt-4o"))
        assert agent

    def test_create_any_with_invalid_string(self) -> None:
        with pytest.raises(ValueError, match="Unsupported agent framework"):
            AnyAgent.create("non-existing", AgentConfig(model_id="gpt-4o"))


# Tests for model arguments
class TestModelArguments:
    def test_model_args(
        self, agent_framework: AgentFramework, mock_litellm_response: Any
    ) -> None:
        if agent_framework == AgentFramework.LLAMA_INDEX:
            pytest.skip(
                "LlamaIndex agent uses a litellm streaming syntax that isn't mockable like the rest of the frameworks"
            )

        agent = create_agent_with_model_args(agent_framework)

        # Patch the appropriate litellm import path for this framework
        import_path = LITELLM_IMPORT_PATHS[agent_framework]
        with patch(import_path) as mock_litellm:
            # Configure the mock
            mock_litellm.return_value = mock_litellm_response

            # Run the agent
            result = agent.run(TEST_QUERY)

            # Verify results
            assert EXPECTED_OUTPUT == result.final_output
            assert mock_litellm.call_args.kwargs["temperature"] == TEST_TEMPERATURE
            assert mock_litellm.call_args.kwargs["frequency_penalty"] == TEST_PENALTY
            assert mock_litellm.call_count > 0

    def test_model_args_streaming(self, agent_framework: AgentFramework) -> None:
        if agent_framework != AgentFramework.LLAMA_INDEX:
            pytest.skip("This test is only for LlamaIndex framework")

        agent = create_agent_with_model_args(agent_framework)

        # Patch the appropriate litellm import path for LlamaIndex
        import_path = LITELLM_IMPORT_PATHS[agent_framework]
        with patch(import_path) as mock_litellm:
            # Configure the mock to return our async generator
            mock_litellm.side_effect = mock_streaming_response

            # Run the agent
            result = agent.run(TEST_QUERY)

            # Verify results
            assert result.final_output
            assert "Harrisburg" in result.final_output
            assert mock_litellm.call_args.kwargs["stream"] is True
            assert mock_litellm.call_args.kwargs["temperature"] == TEST_TEMPERATURE
            assert mock_litellm.call_args.kwargs["frequency_penalty"] == TEST_PENALTY
            assert mock_litellm.call_count > 0
