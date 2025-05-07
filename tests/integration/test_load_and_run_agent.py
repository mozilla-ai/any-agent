import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent, TracingConfig
from any_agent.config import MCPStdio
from any_agent.tracing.trace import AgentTrace, _is_tracing_supported


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
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args: dict[str, Any] = (
        {"parallel_tool_calls": False}
        if agent_framework is not AgentFramework.AGNO
        else {}
    )
    model_args["temperature"] = 0.0
    tools = [
        write_file,
        MCPStdio(
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
    agent = AnyAgent.create(agent_framework, agent_config, tracing=TracingConfig())

    try:
        agent_trace = agent.run(
            "Use the tools to find what year it is in the America/New_York timezone and write the value (single number) to a file",
        )
        assert os.path.exists(os.path.join(tmp_path, tmp_file))
        with open(os.path.join(tmp_path, tmp_file)) as f:
            content = f.read()
        assert content == str(datetime.now().year)
        assert isinstance(agent_trace, AgentTrace)
        assert agent_trace.final_output
        if _is_tracing_supported(agent_framework):
            assert agent_trace.spans
            assert len(agent_trace.spans) > 0
            cost_sum = agent_trace.get_total_cost()
            assert cost_sum.total_cost > 0
            assert cost_sum.total_cost < 1.00
            assert cost_sum.total_tokens > 0
            assert cost_sum.total_tokens < 20000
    finally:
        agent.exit()


def test_run_agent_twice(agent_framework: AgentFramework) -> None:
    """When an agent is run twice, state from the first run shouldn't bleed into the second run"""
    model_id = "gpt-4.1-nano"
    env_check = validate_environment(model_id)
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args: dict[str, Any] = (
        {"parallel_tool_calls": False}
        if agent_framework is not AgentFramework.AGNO
        else {}
    )
    model_args["temperature"] = 0.0
    agent = AnyAgent.create(
        agent_framework,
        AgentConfig(model_id=model_id, model_args=model_args),
    )
    result1 = agent.run("What is the capital of France?")
    result2 = agent.run("What is the capital of Spain?")
    assert result1.final_output != result2.final_output
    if _is_tracing_supported(agent_framework):
        first_spans = result1.spans
        second_spans = result2.spans
        assert second_spans[: len(first_spans)] != first_spans, (
            "Spans from the first run should not be in the second"
        )
