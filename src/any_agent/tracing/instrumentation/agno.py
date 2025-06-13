# mypy: disable-error-code="method-assign,no-untyped-def,union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode

from .common import _set_tool_output

if TYPE_CHECKING:
    from agno.models.message import Message, MessageMetrics
    from agno.models.response import ModelResponse
    from opentelemetry.trace import Span

    from any_agent.frameworks.agno import AgnoAgent


def _set_llm_input(messages: list[Message], span: Span) -> None:
    if not messages:
        return
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            default=str,
            ensure_ascii=False,
        ),
    )


def _set_llm_output(assistant_message: Message, span: Span) -> None:
    if content := getattr(assistant_message, "content", None):
        span.set_attributes(
            {
                "gen_ai.output": str(content),
                "gen_ai.output.type": "text",
            }
        )
    if tool_calls := getattr(assistant_message, "tool_calls", None):
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": tool.get("function", {}).get(
                                "name", "No name"
                            ),
                            "tool.args": tool.get("function", {}).get(
                                "arguments", "No args"
                            ),
                        }
                        for tool in tool_calls
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )
    metrics: MessageMetrics | None
    if metrics := getattr(assistant_message, "metrics", None):
        span.set_attributes(
            {
                "gen_ai.usage.input_tokens": metrics.input_tokens,
                "gen_ai.usage.output_tokens": metrics.output_tokens,
            }
        )


class _AgnoInstrumentor:
    def __init__(self) -> None:
        self._original_aprocess_model: Any = None
        self._original_arun_function_calls: Any = None
        self.first_llm_calls: set[int] = set()

    def instrument(self, agent: AgnoAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        model = agent._agent.model
        tracer = agent._tracer

        self._original_aprocess_model = model._aprocess_model_response

        async def wrap_aprocess_model_response(
            *args,
            **kwargs,
        ) -> tuple[Message, bool]:
            with tracer.start_as_current_span(f"call_llm {model.id}") as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "call_llm",
                        "gen_ai.request.model": model.id,
                    }
                )
                trace_id = span.get_span_context().trace_id
                if trace_id not in self.first_llm_calls:
                    self.first_llm_calls.add(trace_id)
                    _set_llm_input(kwargs.get("messages", []), span)

                assistant_message: Message
                has_tool_calls: bool
                assistant_message, has_tool_calls = await self._original_aprocess_model(
                    *args, **kwargs
                )

                _set_llm_output(assistant_message, span)

                span.set_status(StatusCode.OK)

            agent._running_traces[trace_id].add_span(span)
            return assistant_message, has_tool_calls

        model._aprocess_model_response = wrap_aprocess_model_response

        self._original_arun_function_calls = model.arun_function_calls

        async def wrap_arun_function_calls(*args, **kwargs):
            tool_call_spans = {}
            function_call_response: ModelResponse
            async for function_call_response in self._original_arun_function_calls(
                *args, **kwargs
            ):
                if function_call_response.event == "ToolCallStarted":
                    if tool_executions := function_call_response.tool_executions:
                        tool = function_call_response.tool_executions[0]
                        tool_name = getattr(tool, "tool_name", "No name")
                        tool_args = getattr(tool, "tool_args", {})
                        tool_call_id = getattr(tool, "tool_call_id", "No id")
                        span: Span = tracer.start_span(
                            name=f"execute_tool {tool_name}",
                        )
                        span.set_attributes(
                            {
                                "gen_ai.operation.name": "execute_tool",
                                "gen_ai.tool.name": tool_name,
                                "gen_ai.tool.args": json.dumps(
                                    tool_args,
                                    default=str,
                                    ensure_ascii=False,
                                ),
                                "gen_ai.tool.call.id": tool_call_id,
                            }
                        )
                    tool_call_spans[tool_call_id] = span
                elif function_call_response.event == "ToolCallCompleted":
                    if tool_executions := function_call_response.tool_executions:
                        tool_call_id = getattr(
                            tool_executions[0], "tool_call_id", "No id"
                        )
                        span = tool_call_spans[tool_call_id]
                        _set_tool_output(
                            getattr(tool_executions[0], "result", "{}"), span
                        )

                    span.end()
                    trace_id = span.get_span_context().trace_id
                    agent._running_traces[trace_id].add_span(span)
                yield function_call_response

        model.arun_function_calls = wrap_arun_function_calls

    def uninstrument(self, agent: AgnoAgent):
        if len(agent._running_traces) > 1:
            return
        model = agent._agent.model
        if self._original_aprocess_model is not None:
            model._aprocess_model_response = self._original_aprocess_model
        if self._original_arun_function_calls is not None:
            model.arun_function_calls = self._original_arun_function_calls
