from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from opentelemetry.trace import StatusCode
from wrapt.patches import wrap_function_wrapper

from .common import _set_tool_output

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
    from opentelemetry.trace import Span, Tracer


def _set_llm_input(messages: list[list[BaseMessage]], span: Span) -> None:
    if not messages:
        return
    if not messages[0]:
        return

    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {
                    "role": message.type.replace("human", "user"),
                    "content": message.content,
                }
                for message in messages[0]
            ],
            default=str,
            ensure_ascii=False,
        ),
    )


def _set_llm_output(response: LLMResult, span: Span) -> None:
    if not response.generations:
        return
    if not response.generations[0]:
        return

    generation = response.generations[0][0]

    if text := generation.text:
        span.set_attributes(
            {
                "gen_ai.output": text,
                "gen_ai.output.type": "text",
            }
        )
    if message := getattr(generation, "message", None):
        if tool_calls := getattr(message, "tool_calls", None):
            span.set_attributes(
                {
                    "gen_ai.output": json.dumps(
                        [
                            {
                                "tool.name": tool.get("name", "No name"),
                                "tool.args": tool.get("args", "No args"),
                            }
                            for tool in tool_calls
                        ],
                        default=str,
                        ensure_ascii=False,
                    ),
                    "gen_ai.output.type": "json",
                }
            )

    if llm_output := getattr(response, "llm_output", None):
        if token_usage := llm_output.get("token_usage", None):
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                }
            )


class _LangChainTracingCallback(BaseCallbackHandler):
    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self._current_spans: dict[str, dict[str, Span]] = {
            "model": {},
            "tool": {},
        }
        self.first_llm_calls: set[int] = set()
        super().__init__()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        model = kwargs.get("invocation_params", {}).get("model", "No model")
        span: Span = self.tracer.start_span(
            name=f"call_llm {model}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "call_llm",
                "gen_ai.request.model": model,
            }
        )

        trace_id = span.get_span_context().trace_id
        if trace_id not in self.first_llm_calls:
            self.first_llm_calls.add(trace_id)
            _set_llm_input(messages, span)

        self._current_spans["model"][str(run_id)] = span

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        span: Span = self.tracer.start_span(
            name=f"execute_tool {serialized.get('name')}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": serialized.get("name", "No name"),
                "gen_ai.tool.description": serialized.get(
                    "description", "No description"
                ),
                "gen_ai.tool.args": json.dumps(inputs, default=str, ensure_ascii=False),
            }
        )

        self._current_spans["tool"][str(run_id)] = span

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        span = self._current_spans["model"][str(run_id)]
        _set_llm_output(response, span)
        span.set_status(StatusCode.OK)
        span.end()

        del self._current_spans["model"][str(run_id)]

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        span = self._current_spans["tool"][str(run_id)]

        if content := getattr(output, "content", None):
            _set_tool_output(content, span)
            if tool_call_id := getattr(output, "tool_call_id", None):
                span.set_attribute("gen_ai.tool.call.id", tool_call_id)

        span.set_status(StatusCode.OK)
        span.end()

        del self._current_spans["tool"][str(run_id)]


class _LangChainInstrumentor:
    def __init__(self) -> None:
        self._original_ainvoke = None

    def instrument(self, tracer: Tracer) -> None:
        tracing_callback = _LangChainTracingCallback(tracer)

        self._config_set = False

        async def wrap_ainvoke(  # type: ignore[no-untyped-def]
            wrapped: Callable[..., None],
            instance: Any,
            args: Any,
            kwargs: Any,
        ):
            if not self._config_set:
                if "config" in kwargs:
                    if callbacks := kwargs["config"].get("callbacks"):
                        if isinstance(callbacks, list):
                            kwargs["config"]["callbacks"].append(tracing_callback)
                        else:
                            original_callback = kwargs["config"]["callbacks"]
                            kwargs["config"]["callbacks"] = [
                                original_callback,
                                tracing_callback,
                            ]
                    else:
                        kwargs["config"]["callbacks"] = [tracing_callback]
                else:
                    kwargs["config"] = RunnableConfig(callbacks=[tracing_callback])
                self._config_set = True

            return await wrapped(*args, **kwargs)  # type: ignore[func-returns-value]

        import langgraph

        self._original_ainvoke = langgraph.pregel.Pregel.ainvoke  # type: ignore[attr-defined]
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "langgraph.pregel", "Pregel.ainvoke", wrapper=wrap_ainvoke
        )

    def uninstrument(self) -> None:
        if self._original_ainvoke is not None:
            import langgraph

            langgraph.pregel.Pregel.ainvoke = self._original_ainvoke
            self._original_ainvoke = None
