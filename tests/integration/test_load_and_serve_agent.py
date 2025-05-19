import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from litellm.utils import validate_environment

import any_agent.serving
from any_agent import AgentConfig, AgentFramework, AnyAgent, TracingConfig
from any_agent.config import MCPStdio, ServingConfig

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace import ReadableSpan

if any_agent.serving.serving_available:
    from common.client import A2AClient
    from common.types import Message, TextPart

FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT,
    datefmt="[%X]",
)
logger = logging.getLogger("any_agent_test")
logger.setLevel(logging.DEBUG)

"""
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
"""
class SpanNode:
    def __init__(span):
        _span: ReadableSpan = span
        _children: list["SpanNode"] = []
    def add_span(child_span):
        """Add child span sorted according to timestamps"""
        ...

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
@pytest.mark.skipif(
    not any_agent.serving.serving_available,
    reason="Integration tests require the installation of the ADK samples (`pip install 'git+https://github.com/google/A2A#subdirectory=samples/python'`)",
)
@pytest.mark.asyncio
async def test_load_and_serve_agent(
    agent_framework: AgentFramework, tmp_path: Path
) -> None:
    kwargs = {}

    tmp_file = "tmp.txt"
    logger.info("Starting")

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
    test_tracer = InMemorySpanExporter()
    additional_exporters=[test_tracer]
    tracing_config = TracingConfig(additional_exporters=additional_exporters)
    agent_config = AgentConfig(
        tools=tools,  # type: ignore[arg-type]
        instructions="Search the web to answer",
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )
    agent_server = await AnyAgent.create_async(
        agent_framework,
        agent_config,
        tracing=tracing_config,
    )
    try:
        logger.info(f"Agent created: {agent_server}")  # noqa: G004
        async_server = await agent_server.serve(
            ServingConfig(port=5555, endpoint="/test_agent")
        )
        logger.info("Agent serving")

        # TODO use an agent card instead

        # open another call in another thread
        client = A2AClient(url="http://localhost:5555/test_agent")
        result = await client.send_task(
            {
                "id": "1",
                "message": Message(
                    role="user",
                    parts=[
                        TextPart(
                            text="Use the tools to find what year it is in the America/New_York timezone and write the value (single number) to a file"
                        )
                    ],
                ),
            }
        )
        logger.info(f"after sending: {result}")  # noqa: G004
        logger.info(f"after sending (traces):")
        for s in test_tracer.get_finished_spans():  # noqa: G004
            logger.info(s.to_json())
    finally:
        await async_server.shutdown()

    def check_file() -> None:
        assert os.path.exists(os.path.join(tmp_path, tmp_file))
        with open(os.path.join(tmp_path, tmp_file)) as f:
            content = f.read()
        assert content == str(datetime.now().year)

    check_file()
    """
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
    """
