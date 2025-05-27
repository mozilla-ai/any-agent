from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode
from wrapt.patches import resolve_path, wrap_function_wrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    from opentelemetry.trace import Span, Tracer
    from smolagents.agent_types import AgentType
    from smolagents.models import ChatMessage, Model


from .common import _set_tool_output


def _set_llm_input(messages: list[dict[str, Any]], span: Span) -> None:
    if not messages:
        return
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {
                    "role": message["role"].value,
                    "content": message["content"][0]["text"],
                }
                for message in messages
            ],
            default=str,
            ensure_ascii=False,
        ),
    )


def _set_llm_output(response: ChatMessage, span: Span) -> None:
    if content := response.content:
        span.set_attributes(
            {
                "gen_ai.output": content,
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
        self._original_model_calls: dict[str, Callable[..., Any]] = {}
        self._original_tool_call: Callable[..., Any] | None = None

    def instrument(self, tracer: Tracer) -> None:
        def model_call_wrap(wrapped, instance: Model, args, kwargs):  # type: ignore[no-untyped-def]
            with tracer.start_as_current_span(f"call_llm {instance.model_id}") as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "call_llm",
                        "gen_ai.request.model": getattr(
                            instance, "model_id", "No model_id"
                        ),
                    }
                )

                trace_id = span.get_span_context().trace_id
                if trace_id not in self.first_llm_calls:
                    self.first_llm_calls.add(trace_id)
                    _set_llm_input(args[0], span)

                response: ChatMessage = wrapped(*args, **kwargs)

                _set_llm_output(response, span)

                span.set_status(StatusCode.OK)

                return response

        def tool_call_wrap(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
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

                result: AgentType | Any | None = wrapped(*args, **kwargs)

                _set_tool_output(result, span)

                span.set_status(StatusCode.OK)

                return result

        import smolagents

        exported_model_subclasses = [
            attr
            for _, attr in vars(smolagents).items()
            if isinstance(attr, type) and issubclass(attr, smolagents.models.Model)
        ]
        for model_subclass in exported_model_subclasses:
            self._original_model_calls[model_subclass.__name__] = (
                model_subclass.generate
            )
            wrap_function_wrapper(  # type: ignore[no-untyped-call]
                "smolagents.models",
                f"{model_subclass.__name__}.generate",
                wrapper=model_call_wrap,
            )

        self._original_tool_call = smolagents.tools.Tool.__call__
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "smolagents.tools", "Tool.__call__", wrapper=tool_call_wrap
        )

    def uninstrument(self) -> None:
        for model_subclass, original_model_call in self._original_model_calls.items():
            model = resolve_path("smolagents.models", model_subclass)[2]  # type: ignore[no-untyped-call]
            model.generate = original_model_call
        if self._original_tool_call is not None:
            tool = resolve_path("smolagents.tools", "Tool")[2]  # type: ignore[no-untyped-call]
            tool.__call__ = self._original_tool_call
