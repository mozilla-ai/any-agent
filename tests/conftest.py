import json
import logging
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path
from textwrap import dedent
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import ModelResponse

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


@pytest.fixture(
    scope="session"
)  # This means it only gets created once per test session
async def echo_sse_server() -> AsyncGenerator[dict[str, str], None]:
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


@pytest.fixture(autouse=True, scope="session")
def configure_logging(pytestconfig: pytest.Config) -> None:
    """Configure the logging level based on the verbosity of the test run.
    This is a session fixture, so it only gets called once per test session.
    """
    verbosity = pytestconfig.getoption("verbose")
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    setup_logger(level=level)


@pytest.fixture
def mock_litellm_response() -> ModelResponse:
    """Fixture to create a standard mock LiteLLM response"""
    return ModelResponse.model_validate_json(
        '{"id":"chatcmpl-BWnfbHWPsQp05roQ06LAD1mZ9tOjT","created":1747157127,"model":"mistral-small-latest","object":"chat.completion","system_fingerprint":"fp_f5bdcc3276","choices":[{"finish_reason":"stop","index":0,"message":{"content":"The state capital of Pennsylvania is Harrisburg.","role":"assistant","tool_calls":null,"function_call":null,"annotations":[]}}],"usage":{"completion_tokens":11,"prompt_tokens":138,"total_tokens":149,"completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,"reasoning_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens_details":{"audio_tokens":0,"cached_tokens":0}},"service_tier":"default"}'
    )


@pytest.fixture
def mock_litellm_tool_call_response() -> ModelResponse:
    """Fixture to create a mock LiteLLM response that includes tool calls"""
    return ModelResponse.model_validate_json(
        '{"id":"chatcmpl-tool-call","created":1747157127,"model":"gpt-4o-2024-08-06","object":"chat.completion","choices":[{"finish_reason":"tool_calls","index":0,"message":{"content":null,"role":"assistant","tool_calls":[{"id":"call_123","type":"function","function":{"name":"search_web","arguments":"{\\"query\\":\\"latest AI developments\\"}"}}]}}],"usage":{"completion_tokens":20,"prompt_tokens":150,"total_tokens":170}}'
    )


@pytest.fixture
def mock_litellm_streaming() -> Callable[..., AsyncGenerator[Any, None]]:
    """
    Create a fixture that returns an async generator function to mock streaming responses.
    This returns a function that can be used as a side_effect.
    """

    async def _mock_streaming_response(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        # First chunk with role
        yield {
            "choices": [
                {
                    "delta": {"role": "assistant", "content": "The state "},
                    "index": 0,
                    "finish_reason": None,
                }
            ]
        }

        # Middle chunks with content
        yield {
            "choices": [
                {"delta": {"content": "capital of "}, "index": 0, "finish_reason": None}
            ]
        }

        yield {
            "choices": [
                {
                    "delta": {"content": "Pennsylvania is "},
                    "index": 0,
                    "finish_reason": None,
                }
            ]
        }

        # Final chunk with finish reason
        yield {
            "choices": [
                {
                    "delta": {"content": "Harrisburg."},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ]
        }

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
