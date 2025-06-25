# mypy: disable-error-code="no-untyped-def,union-attr"
from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode

if TYPE_CHECKING:
    from collections.abc import Callable

    from opentelemetry.trace import Span
    from smolagents.models import ChatMessage

    from any_agent.frameworks.smolagents import SmolagentsAgent


from .common import _set_tool_output


def _set_llm_input(messages: list[ChatMessage], span: Span) -> None:
    if not messages:
        return
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {
                    "role": message.role.value,  # type: ignore[attr-defined]
                    "content": message.content[0]["text"],  # type: ignore[index]
                }
                for message in messages
                if message.content
            ],
            default=str,
            ensure_ascii=False,
        ),
    )


def _set_llm_output(response: ChatMessage, span: Span) -> None:
    if content := response.content:
        span.set_attributes(
            {
                "gen_ai.output": str(content),
                "gen_ai.output.type": "text",
            }
        )
    if tool_calls := response.tool_calls:
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": tool_call.function.name,
                            "tool.args": tool_call.function.arguments,
                        }
                        for tool_call in tool_calls
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )

    if raw := response.raw:
        if token_usage := raw.get("usage", None):
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                }
            )

        if response_model := raw.get("model", None):
            span.set_attribute("gen_ai.response.model", response_model)


class _SmolagentsInstrumentor:
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()
        self._original_generate: Callable[..., Any] | None = None
        self._original_tools: Any | None = None

    def instrument(self, agent: SmolagentsAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        tracer = agent._tracer

        self._original_generate = agent._agent.model.generate

        def wrap_generate(
            messages,
            stop_sequences=None,
            response_format=None,
            tools_to_call_from=None,
            **kwargs,
        ):
            model_id = str(agent._agent.model.model_id)
            with tracer.start_as_current_span(f"call_llm {model_id}") as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "call_llm",
                        "gen_ai.request.model": model_id,
                    }
                )
                trace_id = span.get_span_context().trace_id
                if trace_id not in self.first_llm_calls:
                    self.first_llm_calls.add(trace_id)
                    _set_llm_input(messages, span)

                response: ChatMessage = self._original_generate(  # type: ignore[misc]
                    messages,
                    stop_sequences,
                    response_format,
                    tools_to_call_from,
                    **kwargs,
                )

                _set_llm_output(response, span)

                span.set_status(StatusCode.OK)
                agent._running_traces[trace_id].add_span(span)

                return response

        agent._agent.model.generate = wrap_generate  # type: ignore[method-assign]

        class WrappedToolCall:
            def __init__(self, name, description, original_forward):
                self.name = name
                self.description = description
                self.original_forward = original_forward

            def forward(self, *args, **kwargs):
                with tracer.start_as_current_span(f"execute_tool {self.name}") as span:
                    span.set_attributes(
                        {
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": self.name,
                            "gen_ai.tool.description": self.description,
                            "gen_ai.tool.args": json.dumps(
                                kwargs, default=str, ensure_ascii=False
                            ),
                        }
                    )

                    output = self.original_forward(*args, **kwargs)
                    _set_tool_output(output, span)

                    trace_id = span.get_span_context().trace_id
                    agent._running_traces[trace_id].add_span(span)
                    return output

        self._original_tools = deepcopy(agent._agent.tools)
        wrapped_tools = {}
        for key, tool in agent._agent.tools.items():
            wrapped = WrappedToolCall(tool.name, tool.description, tool.forward)  # type: ignore[no-untyped-call]
            tool.forward = wrapped.forward
            wrapped_tools[key] = tool
        agent._agent.tools = wrapped_tools

    def uninstrument(self, agent: SmolagentsAgent) -> None:
        if len(agent._running_traces) > 1:
            return
        if self._original_generate is not None:
            agent._agent.model.generate = self._original_generate  # type: ignore[method-assign]
        if self._original_tools is not None:
            agent._agent.tools = self._original_tools
