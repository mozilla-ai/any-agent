# mypy: disable-error-code="method-assign,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from any_agent.callbacks.context import Context
    from any_agent.frameworks.llama_index import LlamaIndexAgent


class _LlamaIndexWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original_take_step: Any | None = None
        self._original_acalls: dict[str, Any] = {}

    async def wrap(self, agent: LlamaIndexAgent) -> None:
        self._original_take_step = agent._agent.take_step

        async def wrap_take_step(*args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            context.shared["model_id"] = getattr(agent._agent.llm, "model", "No model")

            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, *args, **kwargs)

            output = await self._original_take_step(  # type: ignore[misc]
                *args, **kwargs
            )

            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, output)

            return output

        # bypass Pydantic validation because _agent is a BaseModel
        agent._agent.model_config["extra"] = "allow"
        agent._agent.take_step = wrap_take_step

        async def wrap_tool_execution(original_call, metadata, *args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            context.shared["metadata"] = metadata

            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context, *args, **kwargs)

            output = await original_call(**kwargs)

            for callback in agent.config.callbacks:
                context = callback.after_tool_execution(context, output)

            return output

        class WrappedAcall:
            def __init__(self, metadata, original_acall):
                self.metadata = metadata
                self.original_acall = original_acall

            async def acall(self, *args, **kwargs):
                return await wrap_tool_execution(
                    self.original_acall, self.metadata, **kwargs
                )

        for tool in agent._agent.tools:
            self._original_acalls[str(tool.metadata.name)] = tool.acall
            wrapped = WrappedAcall(tool.metadata, tool.acall)
            tool.acall = wrapped.acall

    async def unwrap(self, agent: LlamaIndexAgent) -> None:
        if self._original_take_step:
            agent._agent.take_step = self._original_take_step
        if self._original_acalls:
            for tool in agent._agent.tools:
                tool.acall = self._original_acalls[str(tool.metadata.name)]
