import asyncio
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import logging

from concurrent.futures import ThreadPoolExecutor

import pytest
from litellm.utils import validate_environment

from any_agent import AgentConfig, AgentFramework, AnyAgent, TracingConfig
from any_agent.config import MCPStdio, ServingConfig
from any_agent.tracing.trace import AgentTrace, _is_tracing_supported

from queue import Queue

import any_agent.serving
if any_agent.serving.serving_available:
    from common.client import A2AClient
    from common.types import TaskSendParams, Message, TextPart

FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT,
    datefmt="[%X]",
)
logger = logging.getLogger("any_agent_test")
logger.setLevel(logging.DEBUG)

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
async def test_load_and_serve_agent(agent_framework: AgentFramework, tmp_path: Path) -> None:
    kwargs = {}

    tmp_file = "tmp.txt"
    logger.info(f"Starting")

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
    agent_config = AgentConfig(
        tools=tools,  # type: ignore[arg-type]
        instructions="Search the web to answer",
        model_args=model_args,
        **kwargs,  # type: ignore[arg-type]
    )
    agent_server = await AnyAgent.create_async(agent_framework, agent_config, tracing=TracingConfig())
    try:
        logger.info(f"Agent created: {agent_server}")
        async_server = await agent_server.serve(ServingConfig(port=5555,endpoint="/test_agent"))
        logger.info(f"Agent serving")

        # TODO use an agent card instead

        # open another call in another thread
        client = A2AClient(url = "http://localhost:5555/test_agent")
        result = await client.send_task(
            {"id": "1",
            "message": Message(
                role='user',
                parts=[TextPart(text="Use the tools to find what year it is in the America/New_York timezone and write the value (single number) to a file")])
            }
            )
    finally:
        async_server.shutdown()
    
    logger.info(f'after sending: {result}')

    assert os.path.exists(os.path.join(tmp_path, tmp_file))
    with open(os.path.join(tmp_path, tmp_file)) as f:
        content = f.read()
    assert content == str(datetime.now().year)
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
