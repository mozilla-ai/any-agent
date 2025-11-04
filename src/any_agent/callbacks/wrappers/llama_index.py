# mypy: disable-error-code="method-assign,misc,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from any_agent.callbacks.context import Context
    from any_agent.frameworks.llama_index import LlamaIndexAgent


def _import_llama_index_converters() -> tuple[Any, Any]:
    """Import conversion functions from llama_index vendor module."""
    from any_agent.vendor.llama_index_utils import (
        from_openai_message_dict,
        to_openai_message_dicts,
    )

    return to_openai_message_dicts, from_openai_message_dict


class _LlamaIndexWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original_take_step: Any | None = None
        self._original_acalls: dict[str, Any] = {}
        self._original_llm_call: Any | None = None

    async def wrap(self, agent: LlamaIndexAgent) -> None:
        self._original_take_step = agent._agent.take_step

        async def wrap_take_step(*args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            context.shared["model_id"] = getattr(agent._agent.llm, "model", "No model")

            if len(args) > 1 and isinstance(args[1], list):
                to_openai_message_dicts, from_openai_message_dict = (
                    _import_llama_index_converters()
                )

                normalized_messages = to_openai_message_dicts(args[1])
                context.framework_state.messages = normalized_messages

                def get_messages():
                    return context.framework_state.messages

                def set_messages(new_messages):
                    context.framework_state.messages = new_messages
                    args[1].clear()
                    args[1].extend(
                        [from_openai_message_dict(msg) for msg in new_messages]
                    )

                context.framework_state._message_getter = get_messages
                context.framework_state._message_setter = set_messages

            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, *args, **kwargs)

            output = await self._original_take_step(  # type: ignore[misc]
                *args, **kwargs
            )

            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, output)

            context.framework_state._message_getter = None
            context.framework_state._message_setter = None

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

        self._original_llm_call = agent.call_model

        async def wrap_call_model(**kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, **kwargs)

            output = await self._original_llm_call(**kwargs)

            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, output)

            return output

        agent.call_model = wrap_call_model

    async def unwrap(self, agent: LlamaIndexAgent) -> None:
        if self._original_take_step:
            agent._agent.take_step = self._original_take_step
        if self._original_acalls:
            for tool in agent._agent.tools:
                tool.acall = self._original_acalls[str(tool.metadata.name)]
        if self._original_llm_call is not None:
            agent.call_model = self._original_llm_call
