import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from any_agent import AgentConfig, AgentFramework, AnyAgent, TracingConfig
from any_agent.config import MCPStdioParams


def check_uvx_installed() -> bool:
    """The integration tests requires uvx"""
    try:
        result = subprocess.run(  # noqa: S603
            ["uvx", "--version"],  # noqa: S607
            capture_output=True,
            check=True,
        )
        return True if result.returncode == 0 else False  # noqa: TRY300
    except Exception:
        return False


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_agent(agent_framework: AgentFramework, tmp_path: Path) -> None:
    kwargs = {}

    tmp_file = "tmp.txt"

    if not check_uvx_installed():
        msg = "uvx is not installed. Please install it to run this test."
        raise RuntimeError(msg)

    def write_file(text: str) -> None:
        """write the text to a file in the tmp_path directory

        Args:
            text (str): The text to write to the file.

        Returns:
            None
        """
        with open(os.path.join(tmp_path, tmp_file), "w", encoding="utf-8") as f:
            f.write(text)

    kwargs["model_id"] = "gpt-4.1-mini"
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip(f"OPENAI_API_KEY needed for {agent_framework}")
    model_args: dict[str, Any] = (
        {"parallel_tool_calls": False}
        if agent_framework is not AgentFramework.AGNO
        else {}
    )
    model_args["temperature"] = 0.0
    tools = [
        write_file,
        MCPStdioParams(
            command="uvx",
            args=["mcp-server-time", "--local-timezone=America/New_York"],
            tools=[
                "get_current_time",
            ],
        ),
    ]
    agent_config = AgentConfig(
        tools=tools,  # type: ignore[arg-type]
        instructions="Search the web to answer",
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )
    traces = tmp_path / "traces"
    agent = AnyAgent.create(
        agent_framework, agent_config, tracing=TracingConfig(output_dir=str(traces))
    )
    try:
        result = agent.run(
            "Use the tools to find what year is it and write the value (single number) to a file",
        )
        assert os.path.exists(os.path.join(tmp_path, tmp_file))
        with open(os.path.join(tmp_path, tmp_file)) as f:
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
    finally:
        agent.exit()
