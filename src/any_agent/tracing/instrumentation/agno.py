# mypy: disable-error-code="no-untyped-call, no-untyped-def"
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.trace import StatusCode
from wrapt.patches import resolve_path, wrap_function_wrapper

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class _AgnoInstrumentor:
    def instrument(self, tracer: Tracer) -> None:
        async def wrap_aprocess_model_response(
            wrapped,
            instance,
            args,
            kwargs,
        ):
            with tracer.start_as_current_span(f"call_llm {instance.id}") as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "call_llm",
                        "gen_ai.request.model": instance.id,
                    }
                )

                assistant_message, has_tool_calls = await wrapped(*args, **kwargs)

                if assistant_message.content:
                    span.set_attributes(
                        {
                            "genai.output": assistant_message.content,
                            "genai.output.type": "text",
                        }
                    )
                else:
                    span.set_attributes(
                        {
                            "genai.output": json.dumps(
                                assistant_message.tool_calls,
                                default=str,
                                ensure_ascii=False,
                            ),
                            "genai.output.type": "json",
                        }
                    )

                if assistant_message.metrics:
                    span.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": assistant_message.metrics.input_tokens,
                            "gen_ai.usage.output_tokens": assistant_message.metrics.output_tokens,
                        }
                    )

                span.set_status(StatusCode.OK)
                span.end()

            return assistant_message, has_tool_calls

        async def wrap_arun_function_calls(
            wrapped,
            instance,
            args,
            kwargs,
        ):
            tool_call_spans = {}
            async for function_call_response in wrapped(*args, **kwargs):
                if function_call_response.event == "ToolCallStarted":
                    tool = function_call_response.tool_calls[0]
                    span: Span = tracer.start_span(
                        name=f"execute_tool {tool.get('tool_name')}",
                    )
                    span.set_attributes(
                        {
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": tool.get("tool_name"),
                            "gen_ai.tool.args": tool.get("tool_args"),
                            "gen_ai.tool.call.id": tool.get("tool_call_id"),
                        }
                    )
                    tool_call_spans[tool.get("tool_call_id")] = span
                elif function_call_response.event == "ToolCallCompleted":
                    tool = function_call_response.tool_calls[0]
                    span = tool_call_spans[tool.get("tool_call_id")]
                    span.set_attributes(
                        {
                            "genai.output": json.dumps(
                                tool.get("content", {}),
                                default=str,
                                ensure_ascii=False,
                            ),
                            "genai.output.type": "json",
                        }
                    )

                    span.set_status(StatusCode.OK)
                    span.end()

                yield function_call_response

        import agno

        self._original_aprocess_model = agno.models.base.Model._aprocess_model_response  # type: ignore[attr-defined]
        wrap_function_wrapper(
            "agno.models.base",
            "Model._aprocess_model_response",
            wrapper=wrap_aprocess_model_response,
        )

        self._original_arun_function_calls = agno.models.base.Model.arun_function_calls  # type: ignore[attr-defined]
        wrap_function_wrapper(
            "agno.models.base",
            "Model.arun_function_calls",
            wrapper=wrap_arun_function_calls,
        )

    def uninstrument(self) -> None:
        parent = resolve_path("agno.models.base", "Model")[2]
        parent._aprocess_model_response = self._original_aprocess_model
        parent.arun_function_calls = self._original_arun_function_calls
