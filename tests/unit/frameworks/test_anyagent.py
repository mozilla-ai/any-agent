import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import TracingConfig


# Test all supported frameworks
@pytest.mark.parametrize("framework", list(AgentFramework))
def test_load_agent_tracing(tmp_path: Path, framework: AgentFramework) -> None:
    mock_agent = MagicMock(spec=AnyAgent)
    mock_agent.load_agent = AsyncMock(return_value=None)
    mock_agent.trace_filepath = None

    # Dynamically create the import path based on the framework
    agent_class_path = _get_agent_class_path(framework)

    # Skip frameworks that don't support tracing
    if framework in (AgentFramework.AGNO, AgentFramework.GOOGLE):
        return

    with patch(agent_class_path, return_value=mock_agent):
        agent = AnyAgent.create(
            agent_framework=framework,
            agent_config=AgentConfig(
                model_id="gpt-4o",
            ),
            tracing=TracingConfig(output_dir=str(tmp_path)),
        )
        # the agent.trace_filepath should be a file that exists
        assert agent.trace_filepath is not None
        assert os.path.exists(agent.trace_filepath)


@pytest.mark.parametrize("framework", list(AgentFramework))
def test_load_agent_no_tracing(framework: AgentFramework) -> None:
    mock_agent = MagicMock(spec=AnyAgent)
    mock_agent.load_agent = AsyncMock(return_value=None)
    mock_agent.trace_filepath = None

    # Dynamically create the import path based on the framework
    agent_class_path = _get_agent_class_path(framework)

    with patch(agent_class_path, return_value=mock_agent):
        agent = AnyAgent.create(
            agent_framework=framework,
            agent_config=AgentConfig(model_id="gpt-4o"),
        )
        # the agent.trace_filepath should be None
        assert agent.trace_filepath is None


def _get_agent_class_path(framework: AgentFramework) -> str:
    """Helper function to get the import path for an agent class based on framework."""
    framework_map = {
        AgentFramework.SMOLAGENTS: "any_agent.frameworks.smolagents.SmolagentsAgent",
        AgentFramework.LANGCHAIN: "any_agent.frameworks.langchain.LangchainAgent",
        AgentFramework.OPENAI: "any_agent.frameworks.openai.OpenAIAgent",
        AgentFramework.LLAMA_INDEX: "any_agent.frameworks.llama_index.LlamaIndexAgent",
        AgentFramework.GOOGLE: "any_agent.frameworks.google.GoogleAgent",
        AgentFramework.AGNO: "any_agent.frameworks.agno.AgnoAgent",
    }
    return framework_map[framework]
