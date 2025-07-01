# mypy: disable-error-code="method-assign,no-untyped-def,no-untyped-call,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

    from any_agent.frameworks.langchain import LangchainAgent


class _LangChainWrapper:
    def __init__(self) -> None:
        self._original_ainvoke: Any | None = None
        self.context: dict[str, Any] = {}

    async def wrap(self, agent: LangchainAgent) -> None:
        from langchain_core.callbacks.base import BaseCallbackHandler
        from langchain_core.runnables import RunnableConfig

        if len(agent._running_traces) > 1:
            return

        self.context["running_traces"] = agent._running_traces
        self.context["tracer"] = agent._tracer

        def before_llm_call(*args, **kwargs):
            for callback in agent.config.callbacks:
                self.context = callback.before_llm_call(self.context, *args, **kwargs)

        def before_tool_execution(*args, **kwargs):
            for callback in agent.config.callbacks:
                self.context = callback.before_tool_execution(
                    self.context, *args, **kwargs
                )

        def after_llm_call(*args, **kwargs):
            for callback in agent.config.callbacks:
                self.context = callback.after_llm_call(self.context, *args, **kwargs)

        def after_tool_execution(*args, **kwargs):
            for callback in agent.config.callbacks:
                self.context = callback.after_tool_execution(
                    self.context, *args, **kwargs
                )

        class _LangChainTracingCallback(BaseCallbackHandler):
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
                before_llm_call(serialized, messages, **kwargs)

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
                before_tool_execution(serialized, input_str, inputs=inputs, **kwargs)

            def on_llm_end(
                self,
                response: LLMResult,
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                **kwargs: Any,
            ) -> Any:
                after_llm_call(response, **kwargs)

            def on_tool_end(
                self,
                output: Any,
                *,
                run_id: UUID,
                parent_run_id: UUID | None = None,
                **kwargs: Any,
            ) -> Any:
                after_tool_execution(output, **kwargs)

        tracing_callback = _LangChainTracingCallback()

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

    async def unwrap(self, agent: LangchainAgent) -> None:
        if len(agent._running_traces) > 1:
            return
        if self._original_ainvoke is not None:
            agent._agent.ainvoke = self._original_ainvoke
