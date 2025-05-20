from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.trace import StatusCode
from wrapt.patches import wrap_function_wrapper

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer
    from smolagents.agent_types import AgentType
    from smolagents.models import ChatMessage


class _SmolagentsInstrumentor:
    def instrument(self, tracer: Tracer) -> None:
        def model_call_wrap(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span(f"call_llm {instance.model_id}") as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "call_llm",
                        "gen_ai.request.model": instance.model_id,
                    }
                )

                response: ChatMessage = wrapped(*args, **kwargs)

                if response.content:
                    span.set_attributes(
                        {
                            "genai.output": response.content,
                            "genai.output.type": "text",
                        }
                    )
                elif response.tool_calls:
                    tool_calls = [
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                        for tool_call in response.tool_calls
                    ]
                    span.set_attributes(
                        {
                            "genai.output": json.dumps(
                                tool_calls,
                                default=str,
                                ensure_ascii=False,
                            ),
                            "genai.output.type": "json",
                        }
                    )

                if token_usage := getattr(response.raw, "model_extra", {}).get("usage"):
                    if token_usage:
                        span.set_attributes(
                            {
                                "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                                "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                            }
                        )

                if response_model := getattr(response.raw, "model", None):
                    span.set_attribute("gen_ai.response.model", response_model)

                span.set_status(StatusCode.OK)
                span.end()

                return response

        def tool_call_wrap(wrapped, instance, args, kwargs):
            with tracer.start_as_current_span(f"execute_tool {instance.name}") as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": instance.name,
                        "gen_ai.tool.args": json.dumps(
                            kwargs,
                            default=str,
                            ensure_ascii=False,
                        ),
                        "gen_ai.tool.description": instance.description,
                    }
                )

                result: AgentType | None = wrapped(*args, **kwargs)

                if result:
                    span.set_attributes(
                        {
                            "genai.output": result.to_string(),
                            "genai.output.type": "text",
                        }
                    )
                span.set_status(StatusCode.OK)
                span.end()

                return result

        import smolagents

        self._original_model_call = smolagents.models.Model.__call__  # type: ignore[no-untyped-call]
        wrap_function_wrapper(
            "smolagents.models", "Model.__call__", wrapper=model_call_wrap
        )

        self._original_tool_call = smolagents.tools.Tool.__call__  # type: ignore[no-untyped-call]
        wrap_function_wrapper(
            "smolagents.tools", "Tool.__call__", wrapper=tool_call_wrap
        )

    def uninstrument(self) -> None:
        pass
