import time
from collections.abc import AsyncGenerator, Generator
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest
import rich.console
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags, TraceState
from opentelemetry.trace.status import Status, StatusCode

from any_agent.config import AgentFramework


@pytest.fixture(autouse=True)
def disable_rich_console(
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
) -> None:
    original_init = rich.console.Console.__init__

    def quiet_init(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if pytestconfig.option.capture != "no":
            kwargs["quiet"] = True
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(rich.console.Console, "__init__", quiet_init)


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
    await asyncio.sleep(1)

    try:
        yield {"url": "http://127.0.0.1:8000/sse"}
    finally:
        # Clean up the process when test is done
        process.kill()
        await process.wait()
