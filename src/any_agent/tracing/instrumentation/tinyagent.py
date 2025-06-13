# mypy: disable-error-code="method-assign,no-untyped-def"
from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode

from .common import _set_tool_output

if TYPE_CHECKING:
    from collections.abc import Callable

    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        ModelResponse,
        Usage,
    )
    from opentelemetry.trace import Span

    from any_agent.frameworks.tinyagent import TinyAgent


def _set_llm_input(messages: list[dict[str, str]], span: Span) -> None:
    span.set_attribute(
        "gen_ai.input.messages", json.dumps(messages, default=str, ensure_ascii=False)
    )


def _set_llm_output(response: ModelResponse, span: Span) -> None:
    if not response.choices:
        return

    message = getattr(response.choices[0], "message", None)
    if not message:
        return

    if content := getattr(message, "content", None):
        span.set_attributes(
            {
                "gen_ai.output": content,
                "gen_ai.output.type": "text",
            }
        )
    tool_calls: list[ChatCompletionMessageToolCall] | None
    if tool_calls := getattr(message, "tool_calls", None):
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": getattr(tool_call.function, "name", "No name"),
                            "tool.args": getattr(
                                tool_call.function, "arguments", "No name"
                            ),
                        }
                        for tool_call in tool_calls
                        if tool_call.function
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )

    token_usage: Usage | None
    if token_usage := getattr(response, "model_extra", {}).get("usage"):
        if token_usage:
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                }
            )


class _TinyAgentInstrumentor:
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()
        self._original_call_model: Callable[..., Any] | None = None
        self._original_call_tool: Callable[..., Any] | None = None

    def instrument(self, agent: TinyAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        tracer = agent._tracer
        self._original_call_model = agent.call_model

        async def call_model(**kwargs):
            model = kwargs.get("model", "No model")
            with tracer.start_as_current_span(f"call_llm {model}") as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "call_llm",
                        "gen_ai.request.model": model,
                    }
                )
                trace_id = span.get_span_context().trace_id
                if trace_id not in self.first_llm_calls:
                    self.first_llm_calls.add(trace_id)
                    _set_llm_input(kwargs["messages"], span)

                response: ModelResponse = await self._original_call_model(**kwargs)  # type: ignore[misc]

                if response_model := getattr(response, "model", None):
                    span.set_attribute("gen_ai.response.model", response_model)

                _set_llm_output(response, span)

                span.set_status(StatusCode.OK)
                agent._running_traces[trace_id].add_span(span)

                return response

        agent.call_model = call_model

        class WrappedCallTool:
            def __init__(self, original_call_tool):
                self.original_call_tool = original_call_tool

            async def call_tool(self, request: dict[str, Any]):
                with tracer.start_as_current_span(
                    f"execute_tool {request.get('name')}"
                ) as span:
                    span.set_attributes(
                        {
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": request.get("name", "No name"),
                            "gen_ai.tool.args": json.dumps(
                                request.get("arguments", {}),
                                default=str,
                                ensure_ascii=False,
                            ),
                        }
                    )

                    result = await self.original_call_tool(request)

                    _set_tool_output(result, span)

                    trace_id = span.get_span_context().trace_id
                    agent._running_traces[trace_id].add_span(span)

                    return result

        self._original_clients = deepcopy(agent.clients)
        wrapped_tools = {}
        for key, tool in agent.clients.items():
            wrapped = WrappedCallTool(tool.call_tool)  # type: ignore[no-untyped-call]
            tool.call_tool = wrapped.call_tool
            wrapped_tools[key] = tool
        agent.clients = wrapped_tools

    def uninstrument(self, agent: TinyAgent) -> None:
        if len(agent._running_traces) > 1:
            return
        if self._original_call_model:
            agent.call_model = self._original_call_model
        if self._original_clients:
            agent.clients = self._original_clients
