import logging
import os
from collections.abc import Callable

import pytest
from litellm.utils import validate_environment
from rich.logging import RichHandler

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import TracingConfig
from any_agent.tools import search_web, visit_webpage
from any_agent.tracing.trace import AgentSpan, AgentTrace, _is_tracing_supported

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("any_agent_test")
logger.setLevel(logging.DEBUG)


CHILD_TAG = "any_agent.children"


def organize(items: list[AgentSpan]) -> None:
    traces = {}
    for trace in items:
        k = trace.context.span_id
        trace.attributes[CHILD_TAG] = {}
        traces[k] = trace
    for trace in items:
        if trace.parent:
            parent_k = trace.parent.span_id
            if parent_k:
                traces[parent_k].attributes[CHILD_TAG][trace.context.span_id] = trace
            else:
                traces[None] = trace
    logger.info(traces[None].model_dump_json(indent=2))


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_multi_agent(
    agent_framework: AgentFramework,
    check_multi_tool_usage: Callable[[list[AgentSpan]], None],
) -> None:
    kwargs = {}

    if agent_framework is AgentFramework.TINYAGENT:
        pytest.skip(
            f"Skipping test for {agent_framework.name} because it does not support multi-agent"
        )

    kwargs["model_id"] = "gpt-4.1-nano"
    agent_model = kwargs["model_id"]
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args = (
        {"parallel_tool_calls": False}
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
        else None
    )
    main_agent = AgentConfig(
        instructions="Use the available tools to complete the task to obtain additional information to answer the query.",
        description="The orchestrator that can use other agents.",
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )

    managed_agents = [
        AgentConfig(
            name="search_web_agent",
            model_id=agent_model,
            description="Agent that can search the web. It can find answers on the web if the query cannot be answered.",
            tools=[search_web],
            model_args=model_args,
        ),
        AgentConfig(
            name="visit_webpage_agent",
            model_id=agent_model,
            description="Agent that can visit webpages",
            tools=[visit_webpage],
            model_args=model_args,
        ),
    ]
    agent = AnyAgent.create(
        agent_framework=agent_framework,
        agent_config=main_agent,
        managed_agents=managed_agents,
        tracing=TracingConfig(console=False, cost_info=True),
    )

    try:
        agent_trace = agent.run(
            "Which LLM agent framework is the most appropriate to execute SQL queries using grammar constrained decoding? I am working on a business environment on my own premises, and I would prefer hosting an open source model myself."
        )

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
            traces = agent_trace.spans
            organize(traces)
            if agent_framework == AgentFramework.AGNO:
                check_multi_tool_usage(traces)
            else:
                logger.warning(
                    "See https://github.com/mozilla-ai/any-agent/issues/256, multi-agent trace checks not working"
                )
    finally:
        agent.exit()
