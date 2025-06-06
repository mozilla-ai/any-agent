import asyncio
import datetime
import logging
import os
from multiprocessing import Process

import pytest
from litellm.utils import validate_environment
from rich.logging import RichHandler

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.config import TracingConfig
from any_agent.serving import A2AServingConfig
from any_agent.tools import a2a_tool, a2a_tool_async
from any_agent.tracing.agent_trace import AgentTrace
from tests.conftest import build_tree

from .helpers import wait_for_a2a_server

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("any_agent_test")
logger.setLevel(logging.DEBUG)


def _assert_valid_agent_trace(agent_trace: AgentTrace) -> None:
    """Assert that agent_trace is valid and has final output."""
    assert isinstance(agent_trace, AgentTrace)
    assert agent_trace.final_output


def _assert_contains_current_date_info(final_output: str) -> None:
    """Assert that the final output contains current date and time information."""
    now = datetime.datetime.now()
    assert all(
        [
            str(now.year) in final_output,
            str(now.day) in final_output,
            now.strftime("%B") in final_output,
        ]
    )


def _assert_has_date_agent_tool_call(agent_trace: AgentTrace) -> None:
    """Assert that the agent trace contains a tool execution span for the date agent."""
    assert any(
        span.is_tool_execution()
        and span.attributes.get("gen_ai.tool.name", None) == "call_date_agent"
        for span in agent_trace.spans
    )


DATE_PROMPT = (
    "What date and time is it right now? "
    "In your answer please include the year, month, day, and time. "
    "Example answer could be something like 'Today is December 15, 2024'"
)


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
@pytest.mark.asyncio
async def test_load_and_run_multi_agent_a2a(
    agent_framework: AgentFramework, test_port: int
) -> None:
    """Tests that an agent contacts another using A2A using the adapter tool.

    Note that there is an issue when using Google ADK: https://github.com/google/adk-python/pull/566
    """
    if agent_framework in [
        # async a2a is not supported
        AgentFramework.SMOLAGENTS,
        # spans are not built correctly
        AgentFramework.LLAMA_INDEX,
        # AgentFramework.GOOGLE,
    ]:
        pytest.skip(
            "https://github.com/mozilla-ai/any-agent/issues/357 tracks fixing so these tests can be re-enabled"
        )
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
    served_task = None
    served_server = None

    try:
        tool_agent_endpoint = "tool_agent"

        # DATE AGENT

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
        date_agent = await AnyAgent.create_async(
            agent_framework=agent_framework,
            agent_config=date_agent_cfg,
            tracing=TracingConfig(console=False, cost_info=True),
        )

        # SERVING PROPER

        served_agent = date_agent
        (served_task, served_server) = await served_agent.serve_async(
            serving_config=A2AServingConfig(
                port=test_port,
                endpoint=f"/{tool_agent_endpoint}",
                log_level="info",
            )
        )
        await asyncio.sleep(3)

        server_url = f"http://localhost:{test_port}/{tool_agent_endpoint}"

        # Search agent is ready for card resolution

        logger.info(
            "Setting up agent",
            extra={"endpoint": server_url},
        )

        main_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that can use other agents via tools using the A2A protocol.",
            tools=[await a2a_tool_async(server_url)],
            model_args=model_args,
            **kwargs,  # type: ignore[arg-type]
        )

        main_agent = await AnyAgent.create_async(
            agent_framework=agent_framework,
            agent_config=main_agent_cfg,
            tracing=TracingConfig(console=False, cost_info=True),
        )

        agent_trace = await main_agent.run_async(DATE_PROMPT)

        _assert_valid_agent_trace(agent_trace)
        _assert_contains_current_date_info(agent_trace.final_output)
        _assert_has_date_agent_tool_call(agent_trace)

        try:
            span_tree = build_tree(agent_trace.spans).model_dump_json(indent=2)
            logger.info("span tree:")
            logger.info(span_tree)
        except KeyError as e:
            pytest.fail(f"The span tree was not built successfully: {e}")

        final_output_log = f"Final output: {agent_trace.final_output}"
        logger.info(final_output_log)

    finally:
        if served_server:
            await served_server.shutdown()
        if served_task:
            served_task.cancel()


def get_datetime() -> str:
    """Return the current date and time"""
    return str(datetime.datetime.now())


def _run_server(agent_framework_str: str, port: int, endpoint: str, model_id: str):
    """Run the server for the sync test. This needs to be defined outside the test function so that it can be run in a separate process."""
    date_agent_description = "Agent that can return the current date."
    date_agent_cfg = AgentConfig(
        instructions="Use the available tools to obtain additional information to answer the query.",
        name="date_agent",
        model_id=model_id,
        description=date_agent_description,
        tools=[get_datetime],
        model_args={"parallel_tool_calls": False}
        if agent_framework_str not in ["agno", "llama_index"]
        else None,
    )

    date_agent = AnyAgent.create(
        agent_framework=AgentFramework.from_string(agent_framework_str),
        agent_config=date_agent_cfg,
        tracing=TracingConfig(console=False, cost_info=True),
    )

    date_agent.serve(
        serving_config=A2AServingConfig(
            port=port,
            endpoint=f"/{endpoint}",
            log_level="info",
        )
    )


@pytest.mark.skipif(
    os.environ.get("ANY_AGENT_INTEGRATION_TESTS", "FALSE").upper() != "TRUE",
    reason="Integration tests require `ANY_AGENT_INTEGRATION_TESTS=TRUE` env var",
)
def test_load_and_run_multi_agent_a2a_sync(
    agent_framework: AgentFramework, test_port: int
) -> None:
    """Tests that an agent contacts another using A2A using the sync adapter tool.

    Note that there is an issue when using Google ADK: https://github.com/google/adk-python/pull/566
    """
    if agent_framework in [
        # async a2a is not supported
        AgentFramework.SMOLAGENTS,
        # spans are not built correctly
        AgentFramework.LLAMA_INDEX,
        # AgentFramework.GOOGLE,
    ]:
        pytest.skip(
            "https://github.com/mozilla-ai/any-agent/issues/357 tracks fixing so these tests can be re-enabled"
        )

    kwargs = {}

    kwargs["model_id"] = "gpt-4.1-nano"
    agent_model = kwargs["model_id"]
    env_check = validate_environment(kwargs["model_id"])
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    server_process = None
    tool_agent_endpoint = "tool_agent_sync"

    try:
        # Start the server in a separate process
        server_process = Process(
            target=_run_server,
            args=(
                agent_framework.value,
                test_port,
                tool_agent_endpoint,
                agent_model,
            ),
        )
        server_process.start()

        server_url = f"http://localhost:{test_port}/{tool_agent_endpoint}"
        wait_for_a2a_server(server_url)

        logger.info(
            "Setting up sync agent",
            extra={"endpoint": f"http://localhost:{test_port}/{tool_agent_endpoint}"},
        )

        # Create main agent using sync methods
        main_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that can use other agents via tools using the A2A protocol (sync version).",
            tools=[a2a_tool(f"http://localhost:{test_port}/{tool_agent_endpoint}")],
            **kwargs,  # type: ignore[arg-type]
        )

        main_agent = AnyAgent.create(
            agent_framework=agent_framework,
            agent_config=main_agent_cfg,
            tracing=TracingConfig(console=False, cost_info=True),
        )

        agent_trace = main_agent.run(DATE_PROMPT)

        _assert_valid_agent_trace(agent_trace)
        _assert_contains_current_date_info(agent_trace.final_output)
        _assert_has_date_agent_tool_call(agent_trace)

    finally:
        if server_process and server_process.is_alive():
            # Send SIGTERM for graceful shutdown
            server_process.terminate()
            server_process.join(timeout=10)
            if server_process.is_alive():
                # Force kill if graceful shutdown failed
                server_process.kill()
                server_process.join()
