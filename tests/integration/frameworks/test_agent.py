import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from any_agent import (
    AgentConfig,
    AgentFramework,
    AnyAgent,
)
from any_agent.callbacks.span_print import ConsolePrintSpan
from any_agent.config import MCPStdio
from any_agent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    group_spans,
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

    agent_invocations, llm_calls, tool_executions = group_spans(agent_trace.spans)

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


class Step(BaseModel):
    number: int
    description: str


class Steps(BaseModel):
    steps: list[Step]


@pytest.mark.parametrize(
    "model_id",
    [
        # Disabling anthropic until output_type can be handled without relying on `response_format`
        # because that is not supported in some providers.
        # "anthropic/claude-3-5-haiku-latest",
        "google/gemini-2.5-flash",
        "huggingface/tgi",  # This is a Qwen/Qwen3-1.7B hosted in https://endpoints.huggingface.co/mozilla-ai/endpoints/dedicated
        "openai/gpt-4.1-nano",
        "xai/grok-3-mini-latest",
        DEFAULT_SMALL_MODEL_ID,
    ],
)
def test_load_and_run_agent(
    model_id: str,
    agent_framework: AgentFramework,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    if (
        model_id != DEFAULT_SMALL_MODEL_ID
        and agent_framework is not AgentFramework.TINYAGENT
    ):
        pytest.skip("We only test multiple providers with TINYAGENT")

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

    model_args = get_default_agent_model_args(agent_framework)

    if "google" in model_id:
        model_args.pop("parallel_tool_calls", None)

    if "huggingface" in model_id:
        model_args.pop("parallel_tool_calls", None)
        model_args["api_base"] = os.environ["HF_ENDPOINT"]

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
        model_id=model_id,
        tools=tools,  # type: ignore[arg-type]
        instructions="Use the available tools to answer.",
        model_args=model_args,
        output_type=Steps,
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
    if model_id not in ("huggingface/tgi", "google/gemini-2.5-flash"):
        assert_cost(agent_trace)
    assert_tokens(agent_trace)

    if update_trace:
        trace_path = Path(__file__).parent.parent / "assets" / agent_framework.name
        with open(f"{trace_path}_trace.json", "w", encoding="utf-8") as f:
            f.write(agent_trace.model_dump_json(indent=2, serialize_as_any=True))
            f.write("\n")
        html_output = console.export_html(inline_styles=True)
        with open(f"{trace_path}_trace.html", "w", encoding="utf-8") as f:
            f.write(html_output.replace("<!DOCTYPE html>", ""))
