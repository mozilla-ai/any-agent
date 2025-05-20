from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.trace import StatusCode
from wrapt.patches import wrap_function_wrapper

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse
    from opentelemetry.trace import Tracer


class _TinyAgentInstrumentor:
    def instrument(self, tracer: Tracer) -> None:
        async def model_call_wrap(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
            with tracer.start_as_current_span(
                f"call_llm {kwargs.get('model')}"
            ) as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "call_llm",
                        "gen_ai.request.model": kwargs.get("model"),
                    }
                )

                result: ModelResponse = await wrapped(*args, **kwargs)

                span.set_attribute("gen_ai.response.model", result.model)

                if message := getattr(result.choices[0], "message", None):
                    if message.content:
                        span.set_attributes(
                            {
                                "genai.output": message.content,
                                "genai.output.type": "text",
                            }
                        )
                    elif message.tool_calls:
                        tool_calls = [
                            {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            }
                            for tool_call in message.tool_calls
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

                if token_usage := getattr(result, "model_extra", {}).get("usage"):
                    if token_usage:
                        span.set_attributes(
                            {
                                "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                                "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                            }
                        )

                span.set_status(StatusCode.OK)
                span.end()

                return result

        async def tool_call_wrap(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
            request = args[0]
            with tracer.start_as_current_span(
                f"execute_tool {request.get('name')}"
            ) as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": request.get("name"),
                        "gen_ai.tool.args": json.dumps(
                            request.get("arguments"),
                            default=str,
                            ensure_ascii=False,
                        ),
                    }
                )

                result = await wrapped(*args, **kwargs)
                span.set_attributes(
                    {
                        "genai.output": json.dumps(
                            result["content"][0]["text"],
                            default=str,
                            ensure_ascii=False,
                        ),
                        "genai.output.type": "json",
                    }
                )

                span.set_status(StatusCode.OK)
                span.end()

                return result

        import litellm

        import any_agent

        self._original_model_call = litellm.acompletion
        wrap_function_wrapper("litellm", "acompletion", wrapper=model_call_wrap)  # type: ignore[no-untyped-call]

        self._original_tool_call = any_agent.frameworks.tinyagent.ToolExecutor.call_tool
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "any_agent.frameworks.tinyagent",
            "ToolExecutor.call_tool",
            wrapper=tool_call_wrap,
        )

    def uninstrument(self) -> None:
        pass
