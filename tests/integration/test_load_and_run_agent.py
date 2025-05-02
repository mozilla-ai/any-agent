import os
from datetime import datetime
from pathlib import Path

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent, TracingConfig
from any_agent.config import MCPStdioParams


def get_current_year() -> str:
    """Get the current year"""
    return str(datetime.now().year)


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_agent(agent_framework: AgentFramework, tmp_path: Path) -> None:
    kwargs = {}

    kwargs["model_id"] = "gpt-4.1-mini"
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip(f"OPENAI_API_KEY needed for {agent_framework}")

    model_args = (
        {"parallel_tool_calls": False}
        if agent_framework is not AgentFramework.AGNO
        else {}
    )
    model_args['temperature'] = 0.0
    tools = [
        get_current_year,
        MCPStdioParams(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "--mount",
                f"type=bind,src={tmp_path},dst=/projects",
                "mcp/filesystem",
                "/projects",
            ],
            tools=[
                "write_file",
            ],
        ),
    ]
    agent_config = AgentConfig(
        tools=tools, # type: ignore[arg-type]
        instructions="Search the web to answer",
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )
    traces = tmp_path / "traces"
    agent = AnyAgent.create(
        agent_framework, agent_config, tracing=TracingConfig(output_dir=str(traces))
    )
    result = agent.run(
        "Use the tools to find what year is it and write the value (single number) to /projects/tmp.txt"
    )
    assert os.path.exists(os.path.join(tmp_path, "tmp.txt"))
    with open(os.path.join(tmp_path, "tmp.txt")) as f:
        content = f.read()
    assert content == str(datetime.now().year)
    assert result
    assert result.final_output
    if agent_framework not in [AgentFramework.LLAMA_INDEX]:
        # Llama Index doesn't currently give back raw_responses.
        assert result.raw_responses
        assert len(result.raw_responses) > 0
    if agent_framework not in (
        AgentFramework.AGNO,
        AgentFramework.GOOGLE,
        AgentFramework.TINYAGENT,
    ):
        assert traces.exists()
        assert agent_framework.name in str(next(traces.iterdir()).name)
        assert result.trace is not None
        assert agent.trace_filepath is not None
        cost_sum = result.trace.get_total_cost()
        assert cost_sum.total_cost > 0
        assert cost_sum.total_cost < 1.00
        assert cost_sum.total_tokens > 0
        assert cost_sum.total_tokens < 20000
