# mypy: disable-error-code="method-assign,misc,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

from any_agent.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

    from any_agent.callbacks.context import Context
    from any_agent.frameworks.langchain import LangchainAgent


def _import_langchain_converters() -> tuple[Any, Any]:
    """Import conversion functions from langchain vendor module."""
    from any_agent.vendor.langchain_any_llm import (
        _convert_dict_to_message,
        _convert_message_to_dict,
    )

    return _convert_dict_to_message, _convert_message_to_dict


class _LangChainWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original_ainvoke: Any | None = None
        self._original_llm_call: Callable[..., Any] | None = None
        self._original_agenerate: Callable[..., Any] | None = None

    async def wrap(self, agent: LangchainAgent) -> None:
        from langchain_core.callbacks.base import BaseCallbackHandler
        from langchain_core.runnables import RunnableConfig

        def before_llm_call(*args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            # Note: Message getters/setters are set up in wrap_agenerate
            # This callback is for span generation via on_chat_model_start
            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, *args, **kwargs)

        def before_tool_execution(*args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context, *args, **kwargs)

        def after_llm_call(*args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, *args, **kwargs)

        def after_tool_execution(*args, **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            for callback in agent.config.callbacks:
                context = callback.after_tool_execution(context, *args, **kwargs)

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

        if agent._model is not None and hasattr(agent._model, "_agenerate"):
            self._original_agenerate = agent._model._agenerate

            async def wrap_agenerate(messages, *args, **kwargs):
                messages_list = list(messages)

                try:
                    context = self.callback_context[
                        get_current_span().get_span_context().trace_id
                    ]
                    _convert_dict_to_message, _convert_message_to_dict = (
                        _import_langchain_converters()
                    )

                    normalized_messages = [
                        _convert_message_to_dict(msg) for msg in messages_list
                    ]
                    context.framework_state.messages = normalized_messages

                    def get_messages():
                        return context.framework_state.messages

                    def set_messages(new_messages):
                        nonlocal messages_list
                        context.framework_state.messages = new_messages
                        messages_list[:] = [
                            _convert_dict_to_message(msg) for msg in new_messages
                        ]

                    context.framework_state._message_getter = get_messages
                    context.framework_state._message_setter = set_messages

                    # Call user callbacks (but not span generation, that will happen in on_chat_model_start)
                    for callback in agent.config.callbacks:
                        if not hasattr(callback, "_set_llm_input"):
                            context = callback.before_llm_call(
                                context, None, [messages_list], **kwargs
                            )

                except Exception:
                    # If we can't get context, just proceed without modification
                    logger.warning(
                        "Could not get context, proceeding without modification"
                    )

                result = await self._original_agenerate(messages_list, *args, **kwargs)

                try:
                    context = self.callback_context[
                        get_current_span().get_span_context().trace_id
                    ]
                    context.framework_state._message_getter = None
                    context.framework_state._message_setter = None
                except Exception:
                    pass

                return result

            agent._model._agenerate = wrap_agenerate
        else:
            logger.warning("Could not wrap _agenerate, proceeding without modification")
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

    async def unwrap(self, agent: LangchainAgent) -> None:
        if self._original_ainvoke is not None:
            agent._agent.ainvoke = self._original_ainvoke

        if (
            self._original_agenerate is not None
            and agent._model is not None
            and hasattr(agent._model, "_agenerate")
        ):
            agent._model._agenerate = self._original_agenerate

        if self._original_llm_call is not None:
            agent.call_model = self._original_llm_call
