from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Status, StatusCode

from any_agent.callbacks.base import Callback

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
                "gen_ai.operation.name": "call_llm",
                "gen_ai.request.model": model_id,
            }
        )

        trace_id = span.get_span_context().trace_id
        if trace_id not in self.first_llm_calls:
            self.first_llm_calls.add(trace_id)
            span.set_attribute(
                "gen_ai.input.messages",
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
                    "gen_ai.output": output,
                    "gen_ai.output.type": "text",
                }
            )
        else:
            span.set_attributes(
                {
                    "gen_ai.output": json.dumps(
                        output,
                        default=str,
                        ensure_ascii=False,
                    ),
                    "gen_ai.output.type": "json",
                }
            )

        span.set_attributes(
            {
                "gen_ai.usage.input_tokens": input_tokens,
                "gen_ai.usage.output_tokens": output_tokens,
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
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": name,
            }
        )
        if description is not None:
            span.set_attribute("gen_ai.tool.description", description)
        if args is not None:
            span.set_attribute(
                "gen_ai.tool.args",
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

        span.set_attributes(
            {"gen_ai.output": tool_output, "gen_ai.output.type": output_type}
        )
        span.set_status(status)

        return context
