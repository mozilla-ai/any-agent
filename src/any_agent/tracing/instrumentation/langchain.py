# mypy: disable-error-code="method-assign, union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import StatusCode

from .common import _set_tool_output

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
    from opentelemetry.trace import Span, Tracer

    from any_agent import AgentTrace
    from any_agent.frameworks.langchain import LangchainAgent


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


class _LangChainInstrumentor:
    def __init__(self) -> None:
        self._original_ainvoke: Any | None = None

    def instrument(self, agent: LangchainAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        from langchain_core.callbacks.base import BaseCallbackHandler
        from langchain_core.runnables import RunnableConfig

        class _LangChainTracingCallback(BaseCallbackHandler):
            def __init__(
                self, tracer: Tracer, running_traces: dict[int, AgentTrace]
            ) -> None:
                self.tracer = tracer
                self.running_traces = running_traces
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
                        "gen_ai.tool.args": json.dumps(
                            inputs, default=str, ensure_ascii=False
                        ),
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

                trace_id = span.get_span_context().trace_id
                self.running_traces[trace_id].add_span(span)
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

                span.end()
                trace_id = span.get_span_context().trace_id
                self.running_traces[trace_id].add_span(span)
                del self._current_spans["tool"][str(run_id)]

        tracing_callback = _LangChainTracingCallback(
            agent._tracer, agent._running_traces
        )

        self._original_ainvoke = agent._agent.ainvoke

        async def wrap_ainvoke(*args, **kwargs):  # type: ignore[no-untyped-def]
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

            return await self._original_ainvoke(*args, **kwargs)  # type: ignore[misc]

        agent._agent.ainvoke = wrap_ainvoke

    def uninstrument(self, agent: LangchainAgent) -> None:
        if len(agent._running_traces) > 1:
            return
        if self._original_ainvoke is not None:
            agent._agent.ainvoke = self._original_ainvoke
