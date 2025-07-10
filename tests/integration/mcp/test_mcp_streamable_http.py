import json
import time
from datetime import datetime, timedelta
from typing import Any

import pytest
from litellm.utils import validate_environment
from pydantic import BaseModel, ConfigDict

from any_agent import (
    AgentConfig,
    AgentFramework,
    AnyAgent,
)
from any_agent.config import MCPStreamableHttp
from any_agent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    group_spans,
)
from any_agent.tracing.agent_trace import AgentSpan, AgentTrace, CostInfo, TokenInfo


def assert_trace(agent_trace: AgentTrace, agent_framework: AgentFramework) -> None:
    def assert_first_llm_call(llm_call: AgentSpan) -> None:
        """Checks the `_set_llm_inputs` implemented by each framework's instrumentation."""
        assert llm_call.attributes.get("gen_ai.input.messages", None) is not None
        # input.messages should be a valid JSON string (list of dicts)
        input_messages_raw = llm_call.attributes.get("gen_ai.input.messages")
        assert input_messages_raw is not None
        input_messages = json.loads(input_messages_raw)
        assert input_messages[0]["role"] == "system"
        assert input_messages[1]["role"] == "user"

    def assert_first_tool_execution(tool_execution: AgentSpan) -> None:
        """Checks the tools setup implemented by each framework's instrumentation."""
        assert tool_execution.attributes.get("gen_ai.tool.args", None) is not None
        # tool.args should be a JSON string (dict)
        tool_args_raw = tool_execution.attributes.get("gen_ai.tool.args")
        assert tool_args_raw is not None
        args = json.loads(tool_args_raw)
        assert "timezone" in args
        assert isinstance(agent_trace, AgentTrace)
        assert agent_trace.final_output

    agent_invocations, llm_calls, tool_executions = group_spans(agent_trace.spans)

    assert len(agent_invocations) == 1

    assert len(llm_calls) >= 2
    assert_first_llm_call(llm_calls[0])

    assert len(tool_executions) >= 1
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
    model_config = ConfigDict(extra="forbid")
    steps: list[Step]


def test_load_and_run_agent_streamable_http(
    agent_framework: AgentFramework,
    request: pytest.FixtureRequest,
    date_streamable_http_server: dict[str, Any],
) -> None:
    kwargs = {}

    kwargs["model_id"] = DEFAULT_SMALL_MODEL_ID
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    tools = [
        MCPStreamableHttp(
            url=date_streamable_http_server["url"],
            client_session_timeout_seconds=30,
        ),
    ]
    agent_config = AgentConfig(
        tools=tools,
        instructions="Use the available tools to answer.",
        model_args=get_default_agent_model_args(agent_framework),
        output_type=Steps,
        **kwargs,  # type: ignore[arg-type]
    )
    agent = AnyAgent.create(agent_framework, agent_config)

    start_ns = time.time_ns()
    agent_trace = agent.run(
        "Return what year it is in the America/New_York timezone. One of the steps returned must include this year."
    )
    end_ns = time.time_ns()

    assert isinstance(agent_trace.final_output, Steps)

    steps = agent_trace.final_output.steps
    assert any(
        str(datetime.now().year) in steps[n].description for n in range(len(steps))
    )

    assert_trace(agent_trace, agent_framework)
    assert_duration(agent_trace, (end_ns - start_ns) / 1_000_000_000)
    assert_cost(agent_trace)
    assert_tokens(agent_trace)
