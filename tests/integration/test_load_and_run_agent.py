import os
from pathlib import Path

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent, TracingConfig
from any_agent.tools import search_web


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
    traces = tmp_path / "traces"
    agent = AnyAgent.create(
        agent_framework, agent_config, tracing=TracingConfig(output_dir=str(traces))
    )
    result = agent.run("Which agent framework is the best?")
    assert result
    if agent_framework not in (AgentFramework.AGNO, AgentFramework.GOOGLE):
        assert traces.exists()
        assert agent_framework.name in str(next(traces.iterdir()).name)
