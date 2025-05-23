import logging
import os
import time
from collections.abc import Callable

import pytest
from litellm.utils import validate_environment
from rich.logging import RichHandler

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import ServingConfig, TracingConfig
from any_agent.tools import a2a_query, search_web
from any_agent.tracing.trace import AgentSpan, AgentTrace

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("any_agent_test")
logger.setLevel(logging.DEBUG)

from threading import Thread

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
    """Tests that an agent contacts another using A2A using the adapter tool.

    Note that there is an issue when using Google ADK: https://github.com/google/adk-python/pull/566
    """
    kwargs = {}

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

    main_agent = None
    served_agent = None

    try:
        tool_agent_port = 5800
        tool_agent_endpoint = "tool_agent"

        search_agent_description = "Agent that can search the web. It can find answers on the web if the query cannot be answered."
        search_agent_cfg = AgentConfig(
            instructions="Use the available tools to complete the task to obtain additional information to answer the query.",
            name="search_web_agent",
            model_id=agent_model,
            description=search_agent_description,
            tools=[search_web],
            model_args=model_args,
        )
        search_agent = AnyAgent.create(
            agent_framework=agent_framework,
            agent_config=search_agent_cfg,
            tracing=TracingConfig(console=False, cost_info=True),
        )

        import datetime
        def get_datetime() -> str:
            """Return the current date and time"""
            return str(datetime.datetime.now())
        date_agent_description = "Agent that can return the current date."
        date_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            name="date_agent",
            model_id=agent_model,
            description=date_agent_description,
            tools=[get_datetime],
            model_args=model_args,
        )
        date_agent = AnyAgent.create(
            agent_framework=agent_framework,
            agent_config=date_agent_cfg,
            tracing=TracingConfig(console=False, cost_info=True),
        )
        

        served_agent = date_agent
        serving_config = ServingConfig(
            port=tool_agent_port, endpoint=f"/{tool_agent_endpoint}"
        )
        thread = Thread(target=lambda: served_agent.serve(serving_config))
        thread.setDaemon(True)
        thread.start()
        while (
            not served_agent._server
            or not served_agent._server.get_uvicorn()
            or not served_agent._server.get_uvicorn().started
        ):
            print("-- waiting for server to start --")
            time.sleep(0.5)

        # Search agent is ready for card resolution

        logger.info("setting up agent in" + f"http://localhost:{tool_agent_port}/{tool_agent_endpoint}")
        main_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that can use other agents via tools using the A2A protocol.",
            tools=[
                a2a_query(
                    f"http://localhost:{tool_agent_port}/{tool_agent_endpoint}"
                )
            ],
            model_args=model_args,
            **kwargs,  # type: ignore[arg-type]
        )

        main_agent = AnyAgent.create(
            agent_framework=agent_framework,
            agent_config=main_agent_cfg,
            tracing=TracingConfig(console=False, cost_info=True),
        )

        agent_trace = main_agent.run(
            # "Which LLM agent framework is the most appropriate to execute SQL queries using grammar constrained decoding? I am working on a business environment on my own premises, and I would prefer hosting an open source model myself."
            "What time is it right now?"
        )

        print(agent_trace.spans)
        print(agent_trace.final_output)

        assert isinstance(agent_trace, AgentTrace)
        assert agent_trace.final_output

        """
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
        """
    finally:
        if served_agent and served_agent._server and served_agent._server.get_uvicorn():
            served_agent._server.get_uvicorn().should_exit = True
        if main_agent:
            main_agent.exit()
        thread.join()
