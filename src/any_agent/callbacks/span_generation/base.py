from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Status, StatusCode

from any_agent.callbacks.base import Callback
from any_agent.tracing.attributes import GenAI

if TYPE_CHECKING:
    from any_agent.callbacks.context import Context


class _SpanGeneration(Callback):
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()

    def _serialize_for_attribute(self, data: Any) -> str:
        """Serialize data for OpenTelemetry attributes, handling various types safely."""
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(data)

    def _determine_output_type(self, output: Any) -> str:
        """Determine output type based on the output content."""
        if isinstance(output, str):
            try:
                json.loads(output)
            except json.JSONDecodeError:
                return "text"
        return "json"

    def _set_llm_input(
        self, context: Context, model_id: str, input_messages: list[dict[str, str]]
    ) -> Context:
        tracer = context.tracer
        span = tracer.start_span(f"call_llm {model_id}")

        span.set_attributes(
            {
                GenAI.OPERATION_NAME: "call_llm",
                GenAI.REQUEST_MODEL: model_id,
            }
        )

        trace_id = span.get_span_context().trace_id
        if trace_id not in self.first_llm_calls:
            self.first_llm_calls.add(trace_id)
            serialized_messages = self._serialize_for_attribute(input_messages)
            span.set_attribute(GenAI.INPUT_MESSAGES, serialized_messages)

        context.current_span = span
        return context

    def _set_llm_output(
        self,
        context: Context,
        output: str | list[dict[str, str]],
        input_tokens: int,
        output_tokens: int,
    ) -> Context:
        span = context.current_span
        output_type = self._determine_output_type(output)
        output_attr = self._serialize_for_attribute(output)

        span.set_attributes(
            {
                GenAI.OUTPUT: output_attr,
                GenAI.OUTPUT_TYPE: output_type,
                GenAI.USAGE_INPUT_TOKENS: input_tokens,
                GenAI.USAGE_OUTPUT_TOKENS: output_tokens,
            }
        )

        span.set_status(StatusCode.OK)
        return context

    def _set_tool_input(
        self,
        context: Context,
        name: str,
        description: str | None = None,
        args: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> Context:
        tracer = context.tracer
        span = tracer.start_span(f"execute_tool {name}")

        attributes = {
            GenAI.OPERATION_NAME: "execute_tool",
            GenAI.TOOL_NAME: name,
        }

        if description is not None:
            attributes[GenAI.TOOL_DESCRIPTION] = description
        if args is not None:
            attributes[GenAI.TOOL_ARGS] = self._serialize_for_attribute(args)
        if call_id is not None:
            attributes["gen_ai.tool.call.id"] = call_id

        span.set_attributes(attributes)
        context.current_span = span
        return context

    def _determine_tool_status(
        self, tool_output: str, output_type: str
    ) -> Status | StatusCode:
        """Determine the status based on tool output content and type."""
        if output_type == "text" and "Error calling tool:" in tool_output:
            return Status(StatusCode.ERROR, description=tool_output)
        return StatusCode.OK

    def _set_tool_output(self, context: Context, tool_output: Any) -> Context:
        span = context.current_span

        if tool_output is None:
            tool_output = "{}"

        output_type = self._determine_output_type(tool_output)
        output_attr = self._serialize_for_attribute(tool_output)
        status = self._determine_tool_status(output_attr, output_type)

        span.set_attributes({GenAI.OUTPUT: output_attr, GenAI.OUTPUT_TYPE: output_type})
        span.set_status(status)
        return context
