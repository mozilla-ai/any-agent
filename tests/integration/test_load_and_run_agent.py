import os
from pathlib import Path

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tools import search_web
from any_agent.tracing import setup_tracing


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_agent(agent_framework: AgentFramework, tmp_path: Path) -> None:
    kwargs = {}

    if agent_framework is AgentFramework.SMOLAGENTS:
        kwargs["agent_type"] = "ToolCallingAgent"

    kwargs["model_id"] = "gpt-4.1-nano"
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip(f"OPENAI_API_KEY needed for {agent_framework}")

    # Agno not yet supported https://github.com/Arize-ai/openinference/issues/1302
    # Google ADK not yet supported https://github.com/Arize-ai/openinference/issues/1506
    if agent_framework not in (AgentFramework.AGNO, AgentFramework.GOOGLE):
        setup_tracing(agent_framework, str(tmp_path / "traces"))

    model_args = (
        {"parallel_tool_calls": False}
        if agent_framework is not AgentFramework.AGNO
        else None
    )

    agent_config = AgentConfig(
        tools=[search_web],
        instructions="Search the web to answer",
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )
    agent = AnyAgent.create(agent_framework, agent_config)
    result = agent.run("Which agent framework is the best?")
    assert len(agent.tools) > 0
    assert result
