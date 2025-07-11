import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from litellm.utils import validate_environment
from pydantic import BaseModel, ConfigDict

from any_agent import (
    AgentConfig,
    AgentFramework,
    AnyAgent,
)
from any_agent.callbacks.span_print import ConsolePrintSpan
from any_agent.config import MCPStdio
from any_agent.evaluation.agent_judge import AgentJudge
from any_agent.evaluation.llm_judge import LlmJudge
from any_agent.evaluation.schemas import EvaluationOutput
from any_agent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
)
from any_agent.tracing.agent_trace import AgentSpan, AgentTrace, CostInfo, TokenInfo
from any_agent.tracing.attributes import GenAI


def uvx_installed() -> bool:
    try:
        result = subprocess.run(
            ["uvx", "--version"],  # noqa: S607
            capture_output=True,
            check=True,
        )
        return True if result.returncode == 0 else False  # noqa: TRY300
    except Exception:
        return False


def assert_trace(agent_trace: AgentTrace, agent_framework: AgentFramework) -> None:
    def assert_first_llm_call(llm_call: AgentSpan) -> None:
        """Checks the `_set_llm_inputs` implemented by each framework's instrumentation."""
        input_messages_raw = llm_call.attributes.get(GenAI.INPUT_MESSAGES)
        assert input_messages_raw is not None
        input_messages = json.loads(input_messages_raw)
        assert input_messages[0]["role"] == "system"
        assert input_messages[1]["role"] == "user"

    def assert_first_tool_execution(tool_execution: AgentSpan) -> None:
        """Checks the tools setup implemented by each framework's instrumentation."""
        assert tool_execution.attributes.get(GenAI.TOOL_ARGS, None) is not None
        # tool.args should be a JSON string (dict)
        tool_args_raw = tool_execution.attributes.get(GenAI.TOOL_ARGS)
        assert tool_args_raw is not None
        args = json.loads(tool_args_raw)
        assert "timezone" in args
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
    assert_first_llm_call(llm_calls[0])

    assert len(tool_executions) >= 2
    assert_first_tool_execution(tool_executions[0])

    messages = agent_trace.spans_to_messages()
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert len(messages) == 2 + len(llm_calls) + len(tool_executions)


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
    """Test evaluation using the new judge classes."""
    # Test 1: Check if agent called write_file tool using LlmJudge
    llm_judge = LlmJudge(
        model_id=DEFAULT_SMALL_MODEL_ID,
        model_args={
            "temperature": 0.0,
        },  # Because it's an llm not agent, the default_model_args are not used
    )
    result1 = llm_judge.run(
        context=str(agent_trace.spans_to_messages()),
        question="Do the messages contain the year 2025?",
    )
    assert isinstance(result1, EvaluationOutput)
    assert result1.passed, (
        f"Expected agent to call write_file tool, but evaluation failed: {result1.reasoning}"
    )

    # Test 2: Check if agent wrote the current year to file using AgentJudge
    agent_judge = AgentJudge(
        model_id=DEFAULT_SMALL_MODEL_ID,
        model_args=get_default_agent_model_args(AgentFramework.TINYAGENT),
    )

    def get_current_year() -> str:
        """Get the current year"""
        return str(datetime.now().year)

    eval_trace = agent_judge.run(
        trace=agent_trace,
        question="Did the agent write the year to a file? Grab the messages from the trace and check if the write_file tool was called.",
        additional_tools=[get_current_year],
    )
    result2 = eval_trace.final_output
    assert isinstance(result2, EvaluationOutput)
    assert result2.passed, (
        f"Expected agent to write current year to file, but evaluation failed: {result2.reasoning}"
    )

    # Test 3: Verify at least one evaluation passes (basic sanity check)
    results = [result1, result2]
    passed_count = sum(1 for r in results if r.passed)
    assert passed_count >= 1, (
        f"Expected at least 1 evaluation to pass, but got {passed_count}/2"
    )


class Step(BaseModel):
    number: int
    description: str


class Steps(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[Step]


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

    kwargs["model_id"] = DEFAULT_SMALL_MODEL_ID
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")
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
        instructions="Use the available tools to answer.",
        model_args=get_default_agent_model_args(agent_framework),
        output_type=Steps,
        **kwargs,  # type: ignore[arg-type]
    )
    agent = AnyAgent.create(agent_framework, agent_config)
    update_trace = request.config.getoption("--update-trace-assets")
    if update_trace:
        for callback in agent.config.callbacks:
            if isinstance(callback, ConsolePrintSpan):
                console = callback.console
                callback.console.record = True

    start_ns = time.time_ns()
    agent_trace = agent.run(
        "Find what year it is in the America/New_York timezone and write the value (single number) to a file. "
        "Finally, return a list of the steps you have taken.",
    )
    end_ns = time.time_ns()

    assert isinstance(agent_trace.final_output, Steps)

    assert (tmp_path / tmp_file).read_text() == str(datetime.now().year)

    assert_trace(agent_trace, agent_framework)
    assert_duration(agent_trace, (end_ns - start_ns) / 1_000_000_000)
    assert_cost(agent_trace)
    assert_tokens(agent_trace)

    if update_trace:
        trace_path = Path(__file__).parent.parent / "assets" / agent_framework.name
        with open(f"{trace_path}_trace.json", "w", encoding="utf-8") as f:
            f.write(agent_trace.model_dump_json(indent=2))
            f.write("\n")
        html_output = console.export_html(inline_styles=True)
        with open(f"{trace_path}_trace.html", "w", encoding="utf-8") as f:
            f.write(html_output.replace("<!DOCTYPE html>", ""))

    assert_eval(agent_trace)
