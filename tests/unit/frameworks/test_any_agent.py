from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.types.utils import ModelResponse

from any_agent import AgentConfig, AgentFramework, AnyAgent


def test_create_any_with_framework(agent_framework: AgentFramework) -> None:
    agent = AnyAgent.create(agent_framework, AgentConfig(model_id="gpt-4o"))
    assert agent


def test_create_any_with_valid_string(agent_framework: AgentFramework) -> None:
    agent = AnyAgent.create(agent_framework.name, AgentConfig(model_id="gpt-4o"))
    assert agent


def test_create_any_with_invalid_string() -> None:
    with pytest.raises(ValueError, match="Unsupported agent framework"):
        AnyAgent.create("non-existing", AgentConfig(model_id="gpt-4o"))


def test_model_args(agent_framework: AgentFramework) -> None:
    if agent_framework == AgentFramework.LLAMA_INDEX:
        pytest.skip(
            "LlamaIndex agent uses a litellm streaming syntax that isn't mockable like the rest of the frameworks"
        )
    create_mock = MagicMock()
    agent_mock = AsyncMock()
    agent_mock.ainvoke.return_value = MagicMock()
    create_mock.return_value = agent_mock
    output = "The state capital of Pennsylvania is Harrisburg."
    test_temp_setting = 0.54321
    test_penalty = 0.5
    agent = AnyAgent.create(
        agent_framework,
        AgentConfig(
            model_id="gpt-4o",
            model_args={
                "temperature": test_temp_setting,
                "frequency_penalty": test_penalty,
            },
        ),
    )

    # Some of the agents have slightly different import path that require different patching
    places_where_litellm_is_imported = {
        AgentFramework.GOOGLE: "google.adk.models.lite_llm.acompletion",
        AgentFramework.LANGCHAIN: "litellm.acompletion",
        AgentFramework.TINYAGENT: "litellm.acompletion",
        AgentFramework.AGNO: "litellm.acompletion",
        AgentFramework.OPENAI: "litellm.acompletion",
        AgentFramework.SMOLAGENTS: "litellm.completion",
    }
    # patch all the imports of the function from the places_where_litellm_is_imported
    with patch(places_where_litellm_is_imported[agent_framework]) as mock_litellm:
        # Make the acompletion function return this response
        mock_litellm.return_value = ModelResponse.model_validate_json(
            '{"id":"chatcmpl-BWnfbHWPsQp05roQ06LAD1mZ9tOjT","created":1747157127,"model":"gpt-4o-2024-08-06","object":"chat.completion","system_fingerprint":"fp_f5bdcc3276","choices":[{"finish_reason":"stop","index":0,"message":{"content":"The state capital of Pennsylvania is Harrisburg.","role":"assistant","tool_calls":null,"function_call":null,"annotations":[]}}],"usage":{"completion_tokens":11,"prompt_tokens":138,"total_tokens":149,"completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,"reasoning_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens_details":{"audio_tokens":0,"cached_tokens":0}},"service_tier":"default"}'
        )

        # Call run which will eventually call _process_single_turn_with_tools
        result = agent.run("what's the state capital of Pennsylvania")
        # Assert that the result contains the expected content
        assert output == result.final_output
        # Verify the mock was called
        assert mock_litellm.call_args.kwargs["temperature"] == test_temp_setting
        assert mock_litellm.call_args.kwargs["frequency_penalty"] == test_penalty
        assert mock_litellm.call_count > 0
