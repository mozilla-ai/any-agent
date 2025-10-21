# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.testing.helpers import LLM_IMPORT_PATHS

TEST_TEMPERATURE = 0.54321
TEST_PENALTY = 0.5
TEST_QUERY = "what's the state capital of Pennsylvania"
EXPECTED_OUTPUT = "The state capital of Pennsylvania is Harrisburg."


def create_agent_with_model_args(framework: AgentFramework) -> AnyAgent:
    """Helper function to create an agent with test model arguments"""
    return AnyAgent.create(
        framework,
        AgentConfig(
            model_id="mistral:mistral-small-latest",
            model_args={
                "temperature": TEST_TEMPERATURE,
                "frequency_penalty": TEST_PENALTY,
            },
        ),
    )


def test_create_any_with_framework(agent_framework: AgentFramework) -> None:
    agent = AnyAgent.create(
        agent_framework, AgentConfig(model_id="mistral:mistral-small-latest")
    )
    assert agent


def test_create_any_with_valid_string(agent_framework: AgentFramework) -> None:
    agent = AnyAgent.create(
        agent_framework.name, AgentConfig(model_id="mistral:mistral-small-latest")
    )
    assert agent


def test_create_any_with_invalid_string() -> None:
    with pytest.raises(ValueError, match="Unsupported agent framework"):
        AnyAgent.create(
            "non-existing", AgentConfig(model_id="mistral:mistral-small-latest")
        )


def test_model_args(
    agent_framework: AgentFramework,
    mock_any_llm_response: Any,
) -> None:
    if agent_framework == AgentFramework.LLAMA_INDEX:
        pytest.skip("LlamaIndex agent uses a any-llm streaming syntax")

    agent = create_agent_with_model_args(agent_framework)

    import_path = LLM_IMPORT_PATHS[agent_framework]

    with patch(import_path, return_value=mock_any_llm_response) as mock_llm:
        result = agent.run(TEST_QUERY)
        assert EXPECTED_OUTPUT == result.final_output
        assert mock_llm.call_args.kwargs["temperature"] == TEST_TEMPERATURE
        assert mock_llm.call_args.kwargs["frequency_penalty"] == TEST_PENALTY
        assert mock_llm.call_count > 0


def test_model_args_streaming(
    agent_framework: AgentFramework,
    mock_any_llm_streaming: Any,
) -> None:
    if agent_framework != AgentFramework.LLAMA_INDEX:
        pytest.skip("This test is only for LlamaIndex framework")

    agent = create_agent_with_model_args(agent_framework)

    # Patch the appropriate import path for LlamaIndex
    import_path = LLM_IMPORT_PATHS[agent_framework]
    mock_streaming = mock_any_llm_streaming

    with patch(import_path, side_effect=mock_streaming) as mock_llm:
        # Run the agent
        result = agent.run(TEST_QUERY)

        # Verify results
        assert result.final_output
        assert "Harrisburg" in result.final_output
        assert mock_llm.call_args.kwargs["stream"] is True
        assert mock_llm.call_args.kwargs["temperature"] == TEST_TEMPERATURE
        assert mock_llm.call_args.kwargs["frequency_penalty"] == TEST_PENALTY
        assert mock_llm.call_count > 0


@pytest.mark.asyncio
async def test_create_sync_in_async_context() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"Cannot use the `sync` API in an `async` context. Use the `async` API instead",
    ):
        AnyAgent.create(
            AgentFramework.TINYAGENT,
            AgentConfig(model_id="mistral:mistral-small-latest"),
        )


@pytest.mark.asyncio
async def test_run_sync_in_async_context() -> None:
    agent = await AnyAgent.create_async(
        AgentFramework.TINYAGENT, AgentConfig(model_id="mistral:mistral-small-latest")
    )
    with pytest.raises(
        RuntimeError,
        match=r"Cannot use the `sync` API in an `async` context. Use the `async` API instead",
    ):
        agent.run(TEST_QUERY)


@pytest.mark.asyncio
async def test_cleanup_async_disconnects_mcp_clients() -> None:
    mock_mcp_client = Mock()
    mock_mcp_client.disconnect = AsyncMock()

    agent = await AnyAgent.create_async(
        AgentFramework.TINYAGENT,
        AgentConfig(
            model_id="mistral:mistral-small-latest",
        ),
    )

    agent._mcp_clients.append(mock_mcp_client)

    assert len(agent._mcp_clients) == 1
    assert agent._mcp_clients[0] == mock_mcp_client

    await agent.cleanup_async()

    mock_mcp_client.disconnect.assert_called_once()
    assert len(agent._mcp_clients) == 0


@pytest.mark.asyncio
async def test_context_manager_automatically_cleans_up() -> None:
    mock_mcp_client = Mock()
    mock_mcp_client.disconnect = AsyncMock()

    async with await AnyAgent.create_async(
        AgentFramework.TINYAGENT,
        AgentConfig(
            model_id="mistral:mistral-small-latest",
        ),
    ) as agent:
        agent._mcp_clients.append(mock_mcp_client)

        assert len(agent._mcp_clients) == 1
        assert agent._mcp_clients[0] == mock_mcp_client
        mock_mcp_client.disconnect.assert_not_called()

    mock_mcp_client.disconnect.assert_called_once()
    assert len(agent._mcp_clients) == 0
