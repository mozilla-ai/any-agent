# mypy: disable-error-code="method-assign,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llama_index.core.agent.workflow.workflow_events import AgentOutput
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.memory import BaseMemory
    from llama_index.core.tools.types import AsyncBaseTool
    from llama_index.core.workflow import Context

    from any_agent.frameworks.llama_index import LlamaIndexAgent


class _LlamaIndexWrapper:
    def __init__(self) -> None:
        self._original_take_step: Any | None = None
        self._original_acalls: dict[str, Any] = {}
        self.context: dict[str, Any] = {}

    async def wrap(self, agent: LlamaIndexAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        self.context["running_traces"] = agent._running_traces
        self.context["tracer"] = agent._tracer
        self.context["model_id"] = getattr(agent._agent.llm, "model", "No model")

        self._original_take_step = agent._agent.take_step

        async def wrap_take_step(
            ctx: Context,
            llm_input: list[ChatMessage],
            tools: Sequence[AsyncBaseTool],
            memory: BaseMemory,
        ) -> AgentOutput:
            for callback in agent.config.callbacks:
                self.context = callback.before_llm_call(
                    self.context, ctx, llm_input, tools, memory
                )

            output: AgentOutput = await self._original_take_step(  # type: ignore[misc]
                ctx, llm_input, tools, memory
            )

            for callback in agent.config.callbacks:
                self.context = callback.after_llm_call(self.context, output)

            return output

        async def tool_execution(original_call, extra_context, *args, **kwargs):
            for k, v in extra_context.items():
                self.context[k] = v

            for callback in agent.config.callbacks:
                self.context = callback.before_tool_execution(
                    self.context, *args, **kwargs
                )

            output = await original_call(**kwargs)

            for callback in agent.config.callbacks:
                self.context = callback.after_tool_execution(
                    self.context, output, *args, **kwargs
                )

            return output

        class WrappedAcall:
            def __init__(self, metadata, original_acall):
                self.metadata = metadata
                self.original_acall = original_acall

            async def acall(self, *args, ctx=None, **kwargs):
                return await tool_execution(
                    self.original_acall, {"tool_name": self.metadata.name}, **kwargs
                )

        for tool in agent._agent.tools:
            self._original_acalls[str(tool.metadata.name)] = tool.acall
            wrapped = WrappedAcall(tool.metadata, tool.acall)
            tool.acall = wrapped.acall

        # bypass Pydantic validation because _agent is a BaseModel
        agent._agent.model_config["extra"] = "allow"
        agent._agent.take_step = wrap_take_step

    async def unwrap(self, agent: LlamaIndexAgent) -> None:
        if len(agent._running_traces) > 1:
            return

        if self._original_take_step:
            agent._agent.take_step = self._original_take_step
        if self._original_acalls:
            for tool in agent._agent.tools:
                tool.acall = self._original_acalls[str(tool.metadata.name)]
