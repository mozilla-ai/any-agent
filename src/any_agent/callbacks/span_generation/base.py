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
            span.set_attribute(
                GenAI.INPUT_MESSAGES,
                json.dumps(
                    input_messages,
                    default=str,
                    ensure_ascii=False,
                ),
            )

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

        if isinstance(output, str):
            span.set_attributes(
                {
                    GenAI.OUTPUT: output,
                    GenAI.OUTPUT_TYPE: "text",
                }
            )
        else:
            span.set_attributes(
                {
                    GenAI.OUTPUT: json.dumps(
                        output,
                        default=str,
                        ensure_ascii=False,
                    ),
                    GenAI.OUTPUT_TYPE: "json",
                }
            )

        span.set_attributes(
            {
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

        span = tracer.start_span(
            name=f"execute_tool {name}",
        )
        span.set_attributes(
            {
                GenAI.OPERATION_NAME: "execute_tool",
                GenAI.TOOL_NAME: name,
            }
        )
        if description is not None:
            span.set_attribute(GenAI.TOOL_DESCRIPTION, description)
        if args is not None:
            span.set_attribute(
                GenAI.TOOL_ARGS,
                json.dumps(
                    args,
                    default=str,
                    ensure_ascii=False,
                ),
            )
        if call_id is not None:
            span.set_attribute("gen_ai.tool.call.id", call_id)

        context.current_span = span

        return context

    def _set_tool_output(self, context: Context, tool_output: Any) -> Context:
        span = context.current_span

        if tool_output is None:
            tool_output = "{}"

        status: Status | StatusCode = StatusCode.OK

        if isinstance(tool_output, str):
            try:
                json.loads(tool_output)
                output_type = "json"
            except json.decoder.JSONDecodeError:
                output_type = "text"
                if "Error calling tool:" in tool_output:
                    status = Status(StatusCode.ERROR, description=tool_output)
        else:
            try:
                tool_output = json.dumps(tool_output, default=str, ensure_ascii=False)
                output_type = "json"
            except TypeError:
                tool_output = str(tool_output)
                output_type = "text"

        span.set_attributes({GenAI.OUTPUT: tool_output, GenAI.OUTPUT_TYPE: output_type})
        span.set_status(status)

        return context
