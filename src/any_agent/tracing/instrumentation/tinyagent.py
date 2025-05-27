from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode
from wrapt.patches import resolve_path, wrap_function_wrapper

from .common import _set_tool_output

if TYPE_CHECKING:
    from collections.abc import Callable

    from litellm.types.utils import ChatCompletionMessageToolCall, ModelResponse, Usage
    from opentelemetry.trace import Span, Tracer


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
        self._original_model_call: Callable[..., Any] | None = None
        self._original_tool_call: Callable[..., Any] | None = None

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
                trace_id = span.get_span_context().trace_id
                if trace_id not in self.first_llm_calls:
                    self.first_llm_calls.add(trace_id)
                    _set_llm_input(kwargs["messages"], span)
                response: ModelResponse = await wrapped(*args, **kwargs)

                span.set_attribute("gen_ai.response.model", response.model)  # type: ignore[arg-type]

                _set_llm_output(response, span)

                span.set_status(StatusCode.OK)

                return response

        async def tool_call_wrap(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
            request: dict[str, Any] = args[0]
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

                result = await wrapped(*args, **kwargs)

                _set_tool_output(result, span)

                span.set_status(StatusCode.OK)

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
        if self._original_model_call:
            import litellm

            litellm.acompletion = self._original_model_call

        if self._original_tool_call:
            parent = resolve_path("any_agent.frameworks.tinyagent", "ToolExecutor")[2]  # type: ignore[no-untyped-call]
            parent.call_tool = self._original_tool_call
