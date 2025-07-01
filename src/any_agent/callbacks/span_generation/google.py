# mypy: disable-error-code="override,union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode, get_current_span

from any_agent.callbacks.base import Callback
from any_agent.callbacks.span_generation.common import _set_tool_output

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext
    from opentelemetry.trace import Span, Tracer


def _set_llm_output(llm_response: LlmResponse, span: Span) -> None:
    content = llm_response.content
    if not content or not content.parts:
        return

    if content.parts[0].text:
        span.set_attributes(
            {
                "gen_ai.output": content.parts[0].text,
                "gen_ai.output.type": "text",
            }
        )
    else:
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": getattr(part.function_call, "name", "No name"),
                            "tool.args": getattr(part.function_call, "args", "{}"),
                        }
                        for part in content.parts
                        if part.function_call
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )


def _set_llm_input(llm_request: LlmRequest, span: Span) -> None:
    if not llm_request.contents:
        return
    messages = []
    if config := llm_request.config:
        messages.append(
            {
                "role": "system",
                "content": getattr(config, "system_instruction", "No instructions"),
            }
        )
    if parts := llm_request.contents[0].parts:
        messages.append(
            {
                "role": getattr(llm_request.contents[0], "role", "No role"),
                "content": getattr(parts[0], "text", "No content"),
            }
        )
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(messages, default=str, ensure_ascii=False),
    )


class _GoogleSpanGeneration(Callback):
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()
        self._current_spans: dict[str, dict[str, Span]] = {
            "model": {},
            "tool": {},
        }
        super().__init__()

    def before_llm_call(
        self,
        context: dict[str, Any],
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> dict[str, Any]:
        tracer: Tracer = context["tracer"]

        span: Span = tracer.start_span(
            name=f"call_llm {llm_request.model}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "call_llm",
                "gen_ai.request.model": llm_request.model or "no_model",
            }
        )
        trace_id = span.get_span_context().trace_id
        if trace_id not in self.first_llm_calls:
            self.first_llm_calls.add(trace_id)
            _set_llm_input(llm_request, span)

        context[f"call_llm-{trace_id}"] = span

        return context

    def after_llm_call(
        self,
        context: dict[str, Any],
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> dict[str, Any]:
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"call_llm-{trace_id}"]
        _set_llm_output(llm_response, span)
        if resp_meta := llm_response.usage_metadata:
            if prompt_tokens := resp_meta.prompt_token_count:
                span.set_attributes({"gen_ai.usage.input_tokens": prompt_tokens})
            if output_tokens := resp_meta.candidates_token_count:
                span.set_attributes({"gen_ai.usage.output_tokens": output_tokens})
        span.set_status(StatusCode.OK)
        return context

    def before_tool_execution(
        self,
        context: dict[str, Any],
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Any | None:
        tracer: Tracer = context["tracer"]

        span: Span = tracer.start_span(
            name=f"execute_tool {tool.name}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool.name,
                "gen_ai.tool.description": tool.description,
                "gen_ai.tool.args": json.dumps(args),
                "gen_ai.tool.call.id": getattr(
                    tool_context, "function_call_id", "no_id"
                ),
            }
        )

        trace_id = span.get_span_context().trace_id
        context[f"execute_tool-{trace_id}"] = span

        return context

    def after_tool_execution(
        self,
        context: dict[str, Any],
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict[Any, Any],
    ) -> Any | None:
        trace_id = get_current_span().get_span_context().trace_id
        span: Span = context[f"execute_tool-{trace_id}"]
        _set_tool_output(tool_response, span)

        return context
