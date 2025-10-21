import json
import logging
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path
from textwrap import dedent
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from any_llm.types.completion import ChatCompletion

from any_agent.config import AgentFramework
from any_agent.logging import setup_logger
from any_agent.testing.helpers import wait_for_server_async
from any_agent.tracing.agent_trace import AgentTrace


@pytest.fixture(params=list(AgentFramework), ids=lambda x: x.name)
def agent_framework(request: pytest.FixtureRequest) -> AgentFramework:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def _patch_stdio_client() -> Generator[
    tuple[AsyncMock, tuple[AsyncMock, AsyncMock]], None, None
]:
    mock_cm = AsyncMock()
    mock_transport = (AsyncMock(), AsyncMock())
    mock_cm.__aenter__.return_value = mock_transport

    with patch("mcp.client.stdio.stdio_client", return_value=mock_cm) as patched:
        yield patched, mock_transport


SSE_MCP_SERVER_SCRIPT = dedent(
    '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server", host="127.0.0.1", port=8000)

        @mcp.tool()
        def write_file(text: str) -> str:
            """Say hi back with the input text"""
            return f"Hi: {text}"

        @mcp.tool()
        def read_file(text: str) -> str:
            """Say bye back the input text"""
            return f"Bye: {text}"

        @mcp.tool()
        def other_tool(text: str) -> str:
            """Say boo back the input text"""
            return f"Boo: {text}"

        mcp.run("sse")
        '''
)


STRHTTP_MCP_SERVER_SCRIPT = dedent(
    '''
        from zoneinfo import ZoneInfo
        from mcp.server.fastmcp import FastMCP
        from mcp.shared.exceptions import McpError
        from datetime import datetime

        def get_zoneinfo(timezone_name: str) -> ZoneInfo:
            try:
                return ZoneInfo(timezone_name)
            except Exception as e:
                msg = "Invalid timezone: " + str(e)
                raise McpError(msg)

        mcp = FastMCP("Dates Server", host="127.0.0.1", port={port})

        @mcp.tool()
        def get_current_time(timezone: str) -> str:
            """Get current time in specified timezone"""
            timezone_info = get_zoneinfo(timezone)
            current_time = datetime.now(timezone_info)

            return(current_time.isoformat(timespec="seconds"))
        mcp.run("streamable-http")
        '''
)


@pytest.fixture(
    scope="session"
)  # This means it only gets created once per test session
async def echo_sse_server() -> AsyncGenerator[dict[str, Any]]:
    """This fixture runs a FastMCP server in a subprocess.
    I thought about trying to mock all the individual mcp client calls,
    but I went with this because this way we don't need to actually mock anything.
    This is similar to what MCPAdapt does in their testing https://github.com/grll/mcpadapt/blob/main/tests/test_core.py
    """
    import asyncio

    process = await asyncio.create_subprocess_exec(
        "python",
        "-c",
        SSE_MCP_SERVER_SCRIPT,
    )

    # Smart ping instead of hardcoded sleep
    await wait_for_server_async("http://127.0.0.1:8000")

    try:
        yield {"url": "http://127.0.0.1:8000/sse"}
    finally:
        # Clean up the process when test is done
        process.kill()
        await process.wait()


@pytest.fixture(scope="session")
async def date_streamable_http_server(worker_id: str) -> AsyncGenerator[dict[str, Any]]:
    """This fixture runs a FastMCP server in a subprocess.
    I thought about trying to mock all the individual mcp client calls,
    but I went with this because this way we don't need to actually mock anything.
    This is similar to what MCPAdapt does in their testing https://github.com/grll/mcpadapt/blob/main/tests/test_core.py
    """
    import asyncio

    port = 19010
    if worker_id and "gw" in worker_id:
        port += int(worker_id.strip("gw"))

    process = await asyncio.create_subprocess_exec(
        "python",
        "-c",
        STRHTTP_MCP_SERVER_SCRIPT.format(port=port),
    )

    # Smart ping instead of hardcoded sleep
    await wait_for_server_async(f"http://127.0.0.1:{port}")

    try:
        yield {"url": f"http://127.0.0.1:{port}/mcp", "port": port}
    finally:
        # Clean up the process when test is done
        process.kill()
        await process.wait()


@pytest.fixture(autouse=True, scope="session")
def configure_logging(pytestconfig: pytest.Config) -> None:
    """Configure the logging level based on the verbosity of the test run.
    This is a session fixture, so it only gets called once per test session.
    """
    verbosity = pytestconfig.getoption("verbose")
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    setup_logger(level=level)


@pytest.fixture
def mock_any_llm_response() -> ChatCompletion:
    """Fixture to create a standard mock any-llm response"""
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-BWnfbHWPsQp05roQ06LAD1mZ9tOjT",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "The state capital of Pennsylvania is Harrisburg.",
                        "role": "assistant",
                    },
                }
            ],
            "created": 1747157127,
            "model": "mistral-small-latest",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 11,
                "prompt_tokens": 138,
                "total_tokens": 149,
            },
        }
    )


@pytest.fixture
def mock_any_llm_tool_call_response() -> ChatCompletion:
    """Fixture to create a mock any-llm response that includes tool calls"""
    return ChatCompletion.model_validate(
        {
            "id": "c98f3cbc69ae4781a863d71b75bcd699",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": "",
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "HscflevQB",
                                "function": {
                                    "arguments": '{"answer": "Hello! How can I assist you today?"}',
                                    "name": "final_answer",
                                },
                                "type": "function",
                            }
                        ],
                    },
                }
            ],
            "created": 1754649356,
            "model": "mistral-small-latest",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 19,
                "prompt_tokens": 84,
                "total_tokens": 103,
            },
        }
    )


@pytest.fixture
def mock_any_llm_streaming() -> Callable[..., AsyncGenerator[Any, None]]:
    """
    Create a fixture that returns an async generator function to mock any-llm streaming responses.
    This returns a function that can be used as a side_effect.
    """

    async def _mock_streaming_response(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        from any_llm.types.completion import ChatCompletionChunk

        yield ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "The state "},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "created": 1747157127,
                "model": "mistral-small-latest",
                "object": "chat.completion.chunk",
            }
        )

        yield ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "delta": {"content": "capital of "},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "created": 1747157127,
                "model": "mistral-small-latest",
                "object": "chat.completion.chunk",
            }
        )

        yield ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "delta": {"content": "Pennsylvania is "},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "created": 1747157127,
                "model": "mistral-small-latest",
                "object": "chat.completion.chunk",
            }
        )

        yield ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "delta": {"content": "Harrisburg."},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "created": 1747157127,
                "model": "mistral-small-latest",
                "object": "chat.completion.chunk",
            }
        )

    return _mock_streaming_response


@pytest.fixture(
    params=list((Path(__file__).parent / "assets").glob("*_trace.json")),
    ids=lambda x: Path(x).stem,
)
def agent_trace(request: pytest.FixtureRequest) -> AgentTrace:
    trace_path = request.param
    with open(trace_path, encoding="utf-8") as f:
        trace = json.load(f)
    return AgentTrace.model_validate(trace)
