from unittest.mock import MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.frameworks.smolagents import AnyLLMModel


def test_load_smolagent_default() -> None:
    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_tool = MagicMock()

    with (
        patch("any_agent.frameworks.smolagents.DEFAULT_AGENT_TYPE", mock_agent),
        patch("any_agent.frameworks.smolagents.DEFAULT_MODEL_TYPE", mock_model),
        patch("smolagents.tool", mock_tool),
    ):
        AnyAgent.create(
            AgentFramework.SMOLAGENTS,
            AgentConfig(
                model_id="openai:o3-mini",
            ),
        )

        mock_agent.assert_called_once_with(
            name="any_agent",
            model=mock_model.return_value,
            verbosity_level=-1,
            tools=[],
        )
        mock_model.assert_called_once_with(
            model_id="openai:o3-mini", api_base=None, api_key=None
        )


def test_load_smolagent_with_api_base() -> None:
    mock_agent = MagicMock()
    mock_model = MagicMock()
    mock_tool = MagicMock()

    with (
        patch("any_agent.frameworks.smolagents.DEFAULT_AGENT_TYPE", mock_agent),
        patch("any_agent.frameworks.smolagents.DEFAULT_MODEL_TYPE", mock_model),
        patch("smolagents.tool", mock_tool),
    ):
        AnyAgent.create(
            AgentFramework.SMOLAGENTS,
            AgentConfig(
                model_id="openai:o3-mini",
                model_args={},
                api_base="https://custom-api.example.com",
            ),
        )

        mock_agent.assert_called_once_with(
            name="any_agent",
            model=mock_model.return_value,
            tools=[],
            verbosity_level=-1,
        )
        mock_model.assert_called_once_with(
            model_id="openai:o3-mini",
            api_base="https://custom-api.example.com",
            api_key=None,
        )


def test_load_smolagents_agent_missing() -> None:
    with patch("any_agent.frameworks.smolagents.smolagents_available", False):
        with pytest.raises(ImportError):
            AnyAgent.create(
                AgentFramework.SMOLAGENTS,
                AgentConfig(model_id="mistral:mistral-small-latest"),
            )


def test_load_smolagent_final_answer() -> None:
    """Regression test for https://github.com/mozilla-ai/any-agent/issues/662"""
    from smolagents import FinalAnswerTool

    mock_model = MagicMock()
    mock_tool = MagicMock()

    with (
        patch("any_agent.frameworks.smolagents.DEFAULT_MODEL_TYPE", mock_model),
        patch("smolagents.tool", mock_tool),
    ):
        agent = AnyAgent.create(
            AgentFramework.SMOLAGENTS,
            AgentConfig(
                model_id="openai:o3-mini",
            ),
        )

        assert isinstance(agent._agent.tools["final_answer"], FinalAnswerTool)  # type: ignore[attr-defined]


def test_run_smolagent_custom_args() -> None:
    mock_agent = MagicMock()
    mock_agent.return_value = MagicMock()
    with (
        patch("any_agent.frameworks.smolagents.DEFAULT_AGENT_TYPE", mock_agent),
        patch("any_agent.frameworks.smolagents.DEFAULT_MODEL_TYPE"),
        patch("smolagents.tool"),
    ):
        agent = AnyAgent.create(
            AgentFramework.SMOLAGENTS,
            AgentConfig(
                model_id="openai:o3-mini",
            ),
        )
        agent.run("foo", max_steps=30)
        mock_agent.return_value.run.assert_called_once_with("foo", max_steps=30)


class TestAnyLLMModel:
    """Tests for AnyLLMModel class directly."""

    def test_malformed_model_id_raises_clear_error(self) -> None:
        """Test that malformed model_id raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Invalid model_id format"):
            AnyLLMModel(model_id="invalid-no-provider")

    def test_parses_model_id_correctly(self) -> None:
        """Test that model_id is parsed into provider and model."""
        with patch("any_llm.AnyLLM.create"):
            model = AnyLLMModel(
                model_id="openai:gpt-4o",
                api_key="test-key",
                api_base="https://api.example.com",
            )

        assert model._provider.value == "openai"
        assert model._anyllm_completion_kwargs["model"] == "gpt-4o"
        assert model._api_key == "test-key"
        assert model._api_base == "https://api.example.com"

    def test_create_client_creates_anyllm_instance(self) -> None:
        """Test that create_client() creates an AnyLLM instance with correct args."""
        mock_anyllm_create = MagicMock()

        with patch("any_llm.AnyLLM.create", mock_anyllm_create):
            model = AnyLLMModel(
                model_id="anthropic:claude-sonnet-4-20250514",
                api_key="test-key",
                api_base="https://api.example.com",
            )

        # Verify create was called with correct provider config.
        mock_anyllm_create.assert_called_with(
            provider=model._provider,
            api_key="test-key",
            api_base="https://api.example.com",
        )

    def test_client_reused_across_multiple_calls(self) -> None:
        """Regression test for GitHub issue #824.

        The original implementation returned the any_llm module from create_client(),
        causing each completion to go through the functional API. This led to
        "Event loop is closed" errors on subsequent calls. The fix creates an
        AnyLLM instance once and reuses it.

        This test verifies:
        1. AnyLLM.create() is called exactly once during initialization
        2. The client instance is stored and reusable for multiple calls
        """
        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_client)

        with patch("any_llm.AnyLLM.create", mock_create):
            model = AnyLLMModel(model_id="openai:gpt-4o")

            # Verify AnyLLM.create was called exactly once during init.
            assert mock_create.call_count == 1

            # Verify the client is the mock we provided.
            assert model.client is mock_client

            # Simulate multiple completion calls (the scenario that caused the bug).
            model.client.completion(messages=[{"role": "user", "content": "Hello"}])
            model.client.completion(messages=[{"role": "user", "content": "World"}])
            model.client.completion(messages=[{"role": "user", "content": "Test"}])

            # Verify all calls went to the same client instance.
            assert mock_client.completion.call_count == 3

            # Verify AnyLLM.create was NOT called again.
            assert mock_create.call_count == 1
