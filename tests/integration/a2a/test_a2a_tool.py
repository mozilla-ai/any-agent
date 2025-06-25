from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any

import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.serving import A2AServingConfig
from any_agent.tools import a2a_tool, a2a_tool_async
from any_agent.tracing.agent_trace import AgentTrace
from tests.integration.helpers import DEFAULT_MODEL_ID, wait_for_server

from .conftest import (
    DATE_PROMPT,
    DEFAULT_TIMEOUT,
    a2a_client_from_agent,
    assert_contains_current_date_info,
    get_datetime,
)

if TYPE_CHECKING:
    from multiprocessing import Queue as QueueType


def _assert_valid_agent_trace(agent_trace: AgentTrace) -> None:
    """Assert that agent_trace is valid and has final output."""
    assert isinstance(agent_trace, AgentTrace)
    assert agent_trace.final_output


def _assert_has_date_agent_tool_call(agent_trace: AgentTrace) -> None:
    """Assert that the agent trace contains a tool execution span for the date agent."""
    assert any(
        span.is_tool_execution()
        and span.attributes.get("gen_ai.tool.name", None) == "call_date_agent"
        for span in agent_trace.spans
    )


@pytest.mark.asyncio
async def test_a2a_tool_async(agent_framework: AgentFramework) -> None:
    """Tests that an agent contacts another using A2A using the adapter tool.

    Note that there is an issue when using Google ADK: https://github.com/google/adk-python/pull/566
    """
    skip_reason = {
        AgentFramework.SMOLAGENTS: "async a2a is not supported",
    }
    if agent_framework in skip_reason:
        pytest.skip(
            f"Framework {agent_framework}, reason: {skip_reason[agent_framework]}"
        )

    env_check = validate_environment(DEFAULT_MODEL_ID)
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    model_args: dict[str, Any] = (
        {"parallel_tool_calls": False}
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
        else {}
    )
    model_args["temperature"] = 0.0

    # Create date agent
    date_agent_cfg = AgentConfig(
        instructions="Use the available tools to obtain additional information to answer the query.",
        name="date_agent",
        model_id=DEFAULT_MODEL_ID,
        description="Agent that can return the current date.",
        tools=[get_datetime],
        model_args=model_args,
    )
    date_agent = await AnyAgent.create_async(
        agent_framework=agent_framework,
        agent_config=date_agent_cfg,
    )

    # Serve the agent and get client
    tool_agent_endpoint = "tool_agent"
    serving_config = A2AServingConfig(
        port=0,
        endpoint=f"/{tool_agent_endpoint}",
        log_level="info",
    )

    async with a2a_client_from_agent(date_agent, serving_config) as (_, server_url):
        # Create main agent with A2A tool
        main_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that can use other agents via tools using the A2A protocol.",
            model_id=DEFAULT_MODEL_ID,
            tools=[
                await a2a_tool_async(
                    server_url, http_kwargs={"timeout": DEFAULT_TIMEOUT}
                )
            ],
            model_args=model_args,
        )

        main_agent = await AnyAgent.create_async(
            agent_framework=agent_framework,
            agent_config=main_agent_cfg,
        )

        agent_trace = await main_agent.run_async(DATE_PROMPT)

        _assert_valid_agent_trace(agent_trace)
        assert_contains_current_date_info(str(agent_trace.final_output))
        _assert_has_date_agent_tool_call(agent_trace)


def _run_server(
    agent_framework_str: str,
    port: int,
    endpoint: str,
    model_id: str,
    server_queue: "QueueType[int]",
) -> None:
    """Run the server for the sync test."""
    date_agent_cfg = AgentConfig(
        instructions="Use the available tools to obtain additional information to answer the query.",
        name="date_agent",
        description="Agent that can return the current date.",
        tools=[get_datetime],
        model_id=model_id,
        model_args={"parallel_tool_calls": False}
        if agent_framework_str not in ["agno", "llama_index"]
        else None,
    )

    date_agent = AnyAgent.create(
        agent_framework=AgentFramework.from_string(agent_framework_str),
        agent_config=date_agent_cfg,
    )

    from any_agent.serving import A2AServingConfig, _get_a2a_app, serve_a2a

    serving_config = A2AServingConfig(
        port=port,
        endpoint=f"/{endpoint}",
        log_level="info",
    )

    app = _get_a2a_app(date_agent, serving_config=serving_config)

    serve_a2a(
        app,
        host=serving_config.host,
        port=serving_config.port,
        endpoint=serving_config.endpoint,
        log_level=serving_config.log_level,
        server_queue=server_queue,
    )


def test_a2a_tool_sync(agent_framework: AgentFramework) -> None:
    """Tests that an agent contacts another using A2A using the sync adapter tool.

    Note that there is an issue when using Google ADK: https://github.com/google/adk-python/pull/566
    """
    skip_reason = {
        AgentFramework.SMOLAGENTS: "async a2a is not supported; run_async_in_sync fails",
    }
    if agent_framework in skip_reason:
        pytest.skip(
            f"Framework {agent_framework}, reason: {skip_reason[agent_framework]}"
        )

    env_check = validate_environment(DEFAULT_MODEL_ID)
    if not env_check["keys_in_environment"]:
        pytest.skip(f"{env_check['missing_keys']} needed for {agent_framework}")

    server_process = None
    tool_agent_endpoint = "tool_agent_sync"

    server_queue = Queue()  # type: ignore[var-annotated]

    try:
        # Start the server in a separate process
        server_process = Process(
            target=_run_server,
            args=(
                agent_framework.value,
                0,
                tool_agent_endpoint,
                DEFAULT_MODEL_ID,
                server_queue,
            ),
        )
        server_process.start()

        test_port = server_queue.get()
        server_url = f"http://localhost:{test_port}/{tool_agent_endpoint}"
        wait_for_server(server_url)

        # Create main agent using sync methods
        main_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that can use other agents via tools using the A2A protocol (sync version).",
            tools=[
                a2a_tool(
                    f"http://localhost:{test_port}/{tool_agent_endpoint}",
                    http_kwargs={"timeout": DEFAULT_TIMEOUT},
                )
            ],
            model_id=DEFAULT_MODEL_ID,
        )

        main_agent = AnyAgent.create(
            agent_framework=agent_framework,
            agent_config=main_agent_cfg,
        )

        agent_trace = main_agent.run(DATE_PROMPT)

        _assert_valid_agent_trace(agent_trace)
        assert_contains_current_date_info(str(agent_trace.final_output))
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
