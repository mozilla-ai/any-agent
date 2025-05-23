import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent, TracingConfig
from any_agent.config import MCPStdio
from any_agent.evaluation import EvaluationCase, evaluate
from any_agent.evaluation.schemas import CheckpointCriteria, TraceEvaluationResult
from any_agent.tracing.agent_trace import AgentTrace, CostInfo, TokenInfo


def uvx_installed() -> bool:
    try:
        result = subprocess.run(  # noqa: S603
            ["uvx", "--version"],  # noqa: S607
            capture_output=True,
            check=True,
        )
        return True if result.returncode == 0 else False  # noqa: TRY300
    except Exception:
        return False


def assert_trace(agent_trace: AgentTrace, agent_framework: AgentFramework) -> None:
    assert isinstance(agent_trace, AgentTrace)
    assert agent_trace.final_output

    agent_invocations = []
    llm_calls = []
    tool_executions = []
    for span in agent_trace.spans:
        if span.is_agent_invocation():
            agent_invocations.append(span)
        elif span.is_llm_call():
            llm_calls.append(span)
        elif span.is_tool_execution():
            tool_executions.append(span)
        else:
            msg = f"Unexpected span: {span}"
            raise AssertionError(msg)

    assert len(agent_invocations) == 1
    assert len(llm_calls) >= 2
    if (
        agent_framework is not AgentFramework.LLAMA_INDEX
    ):  # https://github.com/run-llama/llama_index/issues/18776
        assert len(tool_executions) >= 2


def assert_duration(agent_trace: AgentTrace, wall_time_s: float) -> None:
    assert agent_trace.duration is not None
    assert isinstance(agent_trace.duration, timedelta)
    assert agent_trace.duration.total_seconds() > 0

    diff = abs(agent_trace.duration.total_seconds() - wall_time_s)
    assert diff < 0.1, (
        f"duration ({agent_trace.duration.total_seconds()}s) and wall_time ({wall_time_s}s) differ by more than 0.1s: {diff}s"
    )


def assert_cost(agent_trace: AgentTrace) -> None:
    assert isinstance(agent_trace.cost, CostInfo)
    assert agent_trace.cost.input_cost > 0
    assert agent_trace.cost.output_cost > 0
    assert agent_trace.cost.input_cost + agent_trace.cost.output_cost < 1.00


def assert_tokens(agent_trace: AgentTrace) -> None:
    assert isinstance(agent_trace.tokens, TokenInfo)
    assert agent_trace.tokens.input_tokens > 0
    assert agent_trace.tokens.output_tokens > 0
    assert (agent_trace.tokens.input_tokens + agent_trace.tokens.output_tokens) < 20000


def assert_eval(agent_trace: AgentTrace) -> None:
    case = EvaluationCase(
        llm_judge="gpt-4.1-mini",
        checkpoints=[
            CheckpointCriteria(
                criteria="Check if the agent called the write_file tool and it succeeded",
                points=1,
            ),
            CheckpointCriteria(
                criteria="Check if the agent wrote the year to the file.",
                points=1,
            ),
            CheckpointCriteria(
                criteria="Check if the year was 1990",
                points=1,
            ),
        ],
    )
    result: TraceEvaluationResult = evaluate(
        evaluation_case=case,
        trace=agent_trace,
    )
    assert result
    assert result.score == float(2 / 3)


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_agent(
    agent_framework: AgentFramework, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    kwargs = {}

    tmp_file = "tmp.txt"

    if not uvx_installed():
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
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
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
    update_trace = request.config.getoption("--update-trace-assets")
    if update_trace:
        agent._exporter.console.record = True  # type: ignore[union-attr]

    try:
        start_ns = time.time_ns()
        agent_trace = agent.run(
            "Use the tools to find what year it is in the America/New_York timezone and write the value (single number) to a file",
        )
        end_ns = time.time_ns()

        assert (tmp_path / tmp_file).read_text() == str(datetime.now().year)

        assert_trace(agent_trace, agent_framework)
        assert_duration(agent_trace, (end_ns - start_ns) / 1_000_000_000)
        if (
            agent_framework is not AgentFramework.GOOGLE
        ):  # https://github.com/mozilla-ai/any-agent/issues/287
            assert_cost(agent_trace)
            assert_tokens(agent_trace)
        assert_eval(agent_trace)

        if update_trace:
            trace_path = Path(__file__).parent.parent / "assets" / agent_framework.name
            with open(f"{trace_path}_trace.json", "w", encoding="utf-8") as f:
                f.write(agent_trace.model_dump_json(indent=2))
            html_output = agent._exporter.console.export_html(inline_styles=True)  # type: ignore[union-attr]
            with open(f"{trace_path}_trace.html", "w", encoding="utf-8") as f:
                f.write(html_output.replace("<!DOCTYPE html>", ""))
    finally:
        agent.exit()
