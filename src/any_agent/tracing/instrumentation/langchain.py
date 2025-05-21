from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from opentelemetry.trace import StatusCode
from wrapt.patches import wrap_function_wrapper

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
    from opentelemetry.trace import Span, Tracer


class _LangChainTracingCallback(BaseCallbackHandler):
    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self._current_spans: dict[str, dict[str, Span]] = {
            "model": {},
            "tool": {},
        }
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
        model = kwargs.get("invocation_params", {}).get("model")
        span: Span = self.tracer.start_span(
            name=f"call_llm {model}",
        )
        span.set_attributes(
            {
                "gen_ai.operation.name": "call_llm",
                "gen_ai.request.model": model,
            }
        )

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
                "gen_ai.tool.args": input_str,
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
        generation = response.generations[0][0]

        if generation.text:
            span.set_attributes(
                {
                    "genai.output": generation.text,
                    "genai.output.type": "text",
                }
            )
        else:
            # Tool Call
            span.set_attributes(
                {
                    "genai.output": generation.model_dump_json(exclude_none=True),
                    "genai.output.type": "json",
                }
            )
        if token_usage := getattr(response, "llm_output", {}).get("token_usage"):
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": token_usage.completion_tokens,
                }
            )
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

        if output.content:
            span.set_attributes(
                {
                    "genai.output": json.dumps(
                        output.content,
                        default=str,
                        ensure_ascii=False,
                    ),
                    "genai.output.type": "json",
                    "gen_ai.tool.call.id": output.tool_call_id,
                }
            )

        span.set_status(StatusCode.OK)
        span.end()

        del self._current_spans["tool"][str(run_id)]


class _LangChainInstrumentor:
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
        import langgraph

        langgraph.pregel.Pregel.ainvoke = self._original_ainvoke  # type: ignore[attr-defined]
        self._original_ainvoke = None
