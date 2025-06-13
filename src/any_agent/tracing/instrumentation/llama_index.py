# mypy: disable-error-code="method-assign,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Span, StatusCode

from any_agent.tracing.instrumentation.common import _set_tool_output

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llama_index.core.agent.workflow.workflow_events import AgentOutput
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.llms.llm import ToolSelection
    from llama_index.core.memory import BaseMemory
    from llama_index.core.tools.types import AsyncBaseTool
    from llama_index.core.workflow import Context

    from any_agent.frameworks.llama_index import LlamaIndexAgent


def _set_llm_input(messages: list[ChatMessage], span: Span) -> None:
    span.set_attribute(
        "gen_ai.input.messages",
        json.dumps(
            [
                {
                    "role": message.role.value,
                    "content": message.content or "No content",
                }
                for message in messages
            ]
        ),
    )


def _set_llm_output(output: AgentOutput, span: Span) -> None:
    if response := output.response:
        if content := response.content:
            span.set_attributes(
                {
                    "gen_ai.output": content,
                    "gen_ai.output.type": "text",
                }
            )
    tool_calls: list[ToolSelection]
    if tool_calls := output.tool_calls:
        span.set_attributes(
            {
                "gen_ai.output": json.dumps(
                    [
                        {
                            "tool.name": getattr(tool_call, "tool_name", "No name"),
                            "tool.args": getattr(tool_call, "tool_kwars", "{}"),
                        }
                        for tool_call in tool_calls
                    ],
                    default=str,
                    ensure_ascii=False,
                ),
                "gen_ai.output.type": "json",
            }
        )
    raw: dict[str, Any] | None
    if raw := getattr(output, "raw", None):
        token_usage: dict[str, int]
        if token_usage := raw.get("usage"):
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": token_usage.get("prompt_tokens", 0),
                    "gen_ai.usage.output_tokens": token_usage.get(
                        "completion_tokens", 0
                    ),
                }
            )


class _LlamaIndexInstrumentor:
    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()
        self._original_take_step: Any | None = None
        self._original_acalls: dict[str, Any] = {}

    def instrument(self, agent: LlamaIndexAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        tracer = agent._tracer

        self._original_take_step = agent._agent.take_step

        async def wrap_take_step(
            ctx: Context,
            llm_input: list[ChatMessage],
            tools: Sequence[AsyncBaseTool],
            memory: BaseMemory,
        ) -> AgentOutput:
            model = getattr(agent._agent.llm, "model", "No model")
            with tracer.start_as_current_span(f"call_llm {model}") as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "call_llm",
                        "gen_ai.request.model": model,
                    }
                )
                trace_id = span.get_span_context().trace_id
                if trace_id not in self.first_llm_calls:
                    self.first_llm_calls.add(trace_id)
                    _set_llm_input(llm_input, span)

                output: AgentOutput = await self._original_take_step(  # type: ignore[misc]
                    ctx, llm_input, tools, memory
                )
                if output:
                    _set_llm_output(output, span)

                span.set_status(StatusCode.OK)
                agent._running_traces[trace_id].add_span(span)

            return output

        class WrappedAcall:
            def __init__(self, metadata, original_acall):
                self.metadata = metadata
                self.original_acall = original_acall

            async def acall(self, *args, ctx=None, **kwargs):
                name = self.metadata.name
                with tracer.start_as_current_span(f"execute_tool {name}") as span:
                    span.set_attributes(
                        {
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": name,
                            "gen_ai.tool.args": json.dumps(
                                kwargs, default=str, ensure_ascii=False
                            ),
                        }
                    )

                    output = await self.original_acall(**kwargs)
                    if raw_output := getattr(output, "raw_output", None):
                        if content := getattr(raw_output, "content", None):
                            _set_tool_output(content[0].text, span)
                        else:
                            _set_tool_output(raw_output, span)
                    else:
                        _set_tool_output(output, span)

                    trace_id = span.get_span_context().trace_id
                    agent._running_traces[trace_id].add_span(span)
                    return output

        for tool in agent._agent.tools:
            self._original_acalls[str(tool.metadata.name)] = tool.acall
            wrapped = WrappedAcall(tool.metadata, tool.acall)
            tool.acall = wrapped.acall

        # bypass Pydantic validation because _agent is a BaseModel
        agent._agent.model_config["extra"] = "allow"
        agent._agent.take_step = wrap_take_step

    def uninstrument(self, agent: LlamaIndexAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        if self._original_take_step:
            agent._agent.take_step = self._original_take_step
        if self._original_acalls:
            for tool in agent._agent.tools:
                tool.acall = self._original_acalls[str(tool.metadata.name)]
