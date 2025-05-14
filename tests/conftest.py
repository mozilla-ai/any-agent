import logging
import time
from collections.abc import AsyncGenerator, Callable, Generator
from textwrap import dedent
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from litellm.types.utils import ModelResponse
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags, TraceState
from opentelemetry.trace.status import Status, StatusCode

from any_agent.config import AgentFramework
from any_agent.logging import setup_logger
from any_agent.tracing.trace import AgentSpan


@pytest.fixture
def llm_span() -> ReadableSpan:
    # Convert hex trace and span IDs to integers
    trace_id = int("69ea1b41a9bc5724381993def669c803", 16)
    span_id = int("3817abcfb97cc40c", 16)
    parent_span_id = int("68ff5f13e03ac3fd", 16)

    context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=TraceState(),
    )

    parent = SpanContext(
        trace_id=trace_id,
        span_id=parent_span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=TraceState(),
    )

    resource = Resource.create(
        {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.32.0",
            "service.name": "unknown_service",
        }
    )
    # Create a ReadableSpan
    return ReadableSpan(
        name="ChatLiteLLM",
        context=context,
        kind=SpanKind.INTERNAL,
        parent=parent,
        start_time=int(time.time()),
        end_time=int(time.time() + 1),
        status=Status(StatusCode.OK),
        attributes={
            "any_agent.run_id": str(uuid4()),
            "input.value": '{"messages": [[{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "SystemMessage"], "kwargs": {"content": "Use the available tools to find the answer", "type": "system"}}, {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs": {"content": "Which agent framework is the best?", "type": "human", "id": "2aaf3de6-edf7-4cfa-9483-da348a6749da"}}]]}',
            "input.mime_type": "application/json",
            "output.value": '{"generations": [[{"text": "", "generation_info": {"finish_reason": "tool_calls"}, "type": "ChatGeneration", "message": {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "AIMessage"], "kwargs": {"content": "", "additional_kwargs": {"tool_calls": [{"lc": 1, "type": "not_implemented", "id": ["litellm", "types", "utils", "ChatCompletionMessageToolCall"], "repr": "ChatCompletionMessageToolCall(function=Function(arguments=\'{\\"query\\":\\"best agent frameworks 2023\\"}\', name=\'search_web\'), id=\'call_hYPpXiyNNLiRRytLlJdNFpGN\', type=\'function\')"}]}, "response_metadata": {"token_usage": {"lc": 1, "type": "not_implemented", "id": ["litellm", "types", "utils", "Usage"], "repr": "Usage(completion_tokens=20, prompt_tokens=145, total_tokens=165, completion_tokens_details=CompletionTokensDetailsWrapper(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0, text_tokens=None), prompt_tokens_details=PromptTokensDetailsWrapper(audio_tokens=0, cached_tokens=0, text_tokens=None, image_tokens=None))"}, "model": "gpt-4o-mini", "finish_reason": "tool_calls", "model_name": "gpt-4o-mini"}, "type": "ai", "id": "run-984943f2-6546-47fc-9b1d-81714109e374-0", "tool_calls": [{"name": "search_web", "args": {"query": "best agent frameworks 2023"}, "id": "call_hYPpXiyNNLiRRytLlJdNFpGN", "type": "tool_call"}], "usage_metadata": {"input_tokens": 145, "output_tokens": 20, "total_tokens": 165}, "invalid_tool_calls": []}}}]], "llm_output": {"token_usage": {"completion_tokens": 20, "prompt_tokens": 145, "total_tokens": 165, "completion_tokens_details": {"accepted_prediction_tokens": 0, "audio_tokens": 0, "reasoning_tokens": 0, "rejected_prediction_tokens": 0}, "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0}}, "model": "gpt-4o-mini"}, "run": null, "type": "LLMResult"}',
            "output.mime_type": "application/json",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "Use the available tools to find the answer",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "Which agent framework is the best?",
            "llm.output_messages.0.message.role": "assistant",
            "llm.invocation_parameters": '{"model": "gpt-4o-mini", "temperature": null, "top_p": null, "top_k": null, "n": null, "_type": "litellm-chat", "stop": null, "tools": [{"type": "function", "function": {"name": "search_web", "description": "Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results.\\n\\n    Args:\\n        query (str): The search query to perform.\\n\\n    Returns:\\n        The top search results.", "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"], "type": "object"}}}, {"type": "function", "function": {"name": "visit_webpage", "description": "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages.\\n\\n    Args:\\n        url: The url of the webpage to visit.", "parameters": {"properties": {"url": {"type": "string"}}, "required": ["url"], "type": "object"}}}], "tool_choice": null}',
            "llm.model_name": "gpt-4o-mini",
            "llm.token_count.prompt": 145,
            "llm.token_count.completion": 20,
            "llm.token_count.total": 165,
            "metadata": '{"langgraph_step": 1, "langgraph_node": "agent", "langgraph_triggers": ["branch:to:agent", "start:agent", "tools"], "langgraph_path": ["__pregel_pull", "agent"], "langgraph_checkpoint_ns": "agent:b43d5300-228d-6e29-4b2f-7994f7924340", "checkpoint_ns": "agent:b43d5300-228d-6e29-4b2f-7994f7924340", "ls_provider": "litellm", "ls_model_type": "chat", "ls_model_name": "gpt-4o-mini"}',
            "openinference.span.kind": "LLM",
        },
        events=[],
        links=[],
        resource=resource,
    )


@pytest.fixture(params=list(AgentFramework), ids=lambda x: x.name)
def agent_framework(request: pytest.FixtureRequest) -> AgentFramework:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def _patch_stdio_client() -> Generator[
    tuple[AsyncMock, tuple[AsyncMock, AsyncMock]], None
]:
    mock_cm = AsyncMock()
    mock_transport = (AsyncMock(), AsyncMock())
    mock_cm.__aenter__.return_value = mock_transport

    with patch("mcp.client.stdio.stdio_client", return_value=mock_cm) as patched:
        yield patched, mock_transport


def check_multi_tool_usage_all(json_logs: list[AgentSpan], min_tools: int) -> None:
    tools = len(
        [
            log
            for log in json_logs
            if "openinference.span.kind" in log.attributes
            and log.attributes["openinference.span.kind"] == "TOOL"
        ]
    )
    assert tools < min_tools, (
        "Count of tool usage is too low, managed agents were not used"
    )


check_multi_tool_usage_dict = {
    AgentFramework.GOOGLE: lambda json_logs: check_multi_tool_usage_all(json_logs, 1),
    AgentFramework.LANGCHAIN: lambda json_logs: check_multi_tool_usage_all(
        json_logs, 1
    ),
    AgentFramework.LLAMA_INDEX: lambda json_logs: check_multi_tool_usage_all(
        json_logs, 2
    ),
    AgentFramework.OPENAI: lambda json_logs: check_multi_tool_usage_all(json_logs, 1),
    AgentFramework.AGNO: lambda json_logs: check_multi_tool_usage_all(json_logs, 1),
    AgentFramework.SMOLAGENTS: lambda json_logs: check_multi_tool_usage_all(
        json_logs, 1
    ),
    AgentFramework.TINYAGENT: lambda json_logs: check_multi_tool_usage_all(
        json_logs, 1
    ),
}


@pytest.fixture
def check_multi_tool_usage(
    agent_framework: AgentFramework,
) -> Callable[[list[AgentSpan]], None]:
    return check_multi_tool_usage_dict[agent_framework]


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
async def echo_sse_server() -> AsyncGenerator[dict[str, str]]:
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
    await asyncio.sleep(3)

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
        '{"id":"chatcmpl-BWnfbHWPsQp05roQ06LAD1mZ9tOjT","created":1747157127,"model":"gpt-4o-2024-08-06","object":"chat.completion","system_fingerprint":"fp_f5bdcc3276","choices":[{"finish_reason":"stop","index":0,"message":{"content":"The state capital of Pennsylvania is Harrisburg.","role":"assistant","tool_calls":null,"function_call":null,"annotations":[]}}],"usage":{"completion_tokens":11,"prompt_tokens":138,"total_tokens":149,"completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,"reasoning_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens_details":{"audio_tokens":0,"cached_tokens":0}},"service_tier":"default"}'
    )


@pytest.fixture
def mock_litellm_streaming() -> Callable[[Any, Any], AsyncGenerator[Any, None]]:
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
