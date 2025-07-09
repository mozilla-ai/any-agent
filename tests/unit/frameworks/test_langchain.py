from typing import TYPE_CHECKING
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.testing.helpers import LITELLM_IMPORT_PATHS

if TYPE_CHECKING:
    from any_agent.frameworks.langchain import LangchainAgent


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
        AnyAgent.create(
            AgentFramework.LANGCHAIN,
            AgentConfig(model_id="mistral/mistral-small-latest"),
        )

        model_mock.assert_called_once_with(
            model="mistral/mistral-small-latest",
            api_base=None,
            api_key=None,
            model_kwargs={},
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
            AnyAgent.create(
                AgentFramework.LANGCHAIN,
                AgentConfig(model_id="mistral/mistral-small-latest"),
            )


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
            AgentFramework.LANGCHAIN,
            AgentConfig(model_id="mistral/mistral-small-latest"),
        )
        agent.run("foo", debug=True)
        agent_mock.ainvoke.assert_called_once_with(
            {"messages": [("user", "foo")]}, debug=True, config={"callbacks": [ANY]}
        )


class SampleOutput(BaseModel):
    answer: str
    confidence: float


def test_structured_output_without_tools() -> None:
    """Test that structured output works correctly when no tools are present and tool-related params are not set."""
    config = AgentConfig(model_id="gpt-4.1-mini", output_type=SampleOutput)
    agent: LangchainAgent = AnyAgent.create(AgentFramework.LANGCHAIN, config)  # type: ignore[assignment]

    # Patch the agent's _agent to return a mock result for ainvoke
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [AIMessage(content="Initial response")]
    }
    agent._agent = mock_agent

    def create_mock_response(content: str) -> MagicMock:
        mock_message = MagicMock()
        mock_message.content = content
        mock_message.__getitem__.side_effect = (
            lambda key: content if key == "content" else None
        )
        return MagicMock(choices=[MagicMock(message=mock_message)])

    with patch(LITELLM_IMPORT_PATHS[AgentFramework.LANGCHAIN]) as mock_acompletion:
        mock_acompletion.return_value = create_mock_response(
            '{"answer": "Structured answer", "confidence": 0.95}'
        )
        agent.run("Test question")

        # Only expect that acompletion was called once for structured output
        assert mock_acompletion.call_count == 1
        call_args = mock_acompletion.call_args[1]
        # Should not include any tool-related keys
        assert "tools" not in call_args
        assert "tool_choice" not in call_args
        # Should include response_format
        assert "response_format" in call_args
        assert call_args["response_format"] == SampleOutput
