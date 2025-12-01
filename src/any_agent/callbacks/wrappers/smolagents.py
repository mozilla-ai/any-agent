# mypy: disable-error-code="method-assign,misc,no-untyped-call,no-untyped-def,union-attr"
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from collections.abc import Callable

    from any_agent.callbacks.context import Context
    from any_agent.frameworks.smolagents import SmolagentsAgent


class _SmolagentsWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original_llm_call: Callable[..., Any] | None = None
        self._original_tools: Any | None = None

    async def wrap(self, agent: SmolagentsAgent) -> None:
        try:
            from smolagents.memory import TaskStep
            from smolagents.models import ChatMessage

            smolagents_available = True
        except ImportError:
            smolagents_available = False

        if not smolagents_available:
            msg = "Smolagents is not installed"
            raise ImportError(msg)

        self._original_llm_call = agent._agent.model.generate

        def wrap_generate(messages: list[ChatMessage], **kwargs):
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]
            context.shared["model_id"] = str(agent._agent.model.model_id)

            def get_messages():
                normalized_messages = []
                for msg in messages:
                    msg_dict = msg.dict()

                    # Handle content that might be a list
                    content = msg_dict.get("content")
                    if isinstance(content, list):
                        text_parts = [
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        ]
                        if text_parts and len(text_parts) == len(content):
                            msg_dict["content"] = (
                                " ".join(text_parts)
                                if len(text_parts) > 1
                                else text_parts[0]
                            )

                    normalized_messages.append(msg_dict)
                return normalized_messages

            def set_messages(new_messages):
                messages.clear()
                for msg_dict in new_messages:
                    content = msg_dict["content"]
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]

                    new_msg_dict = {**msg_dict, "content": content}
                    messages.append(ChatMessage.from_dict(new_msg_dict))

                # Update TaskStep in memory so modifications persist through write_memory_to_messages()
                # This is necessary because smolagents rebuilds messages from memory on every LLM call
                if len(new_messages) >= 2:
                    for step in agent._agent.memory.steps:
                        if isinstance(step, TaskStep):
                            # Extract the task text (remove "New task:\n" prefix if present)
                            new_task = new_messages[1]["content"]
                            if isinstance(new_task, str):
                                new_task = new_task.removeprefix("New task:\n")
                            step.task = new_task
                            break

            context.framework_state._message_getter = get_messages
            context.framework_state._message_setter = set_messages

            # Only invoke callbacks on the first LLM call, not on retry attempts
            # Retries can be detected by checking if there are error messages in the history
            is_retry = any(
                "Error:" in str(msg.content) and "Now let's retry" in str(msg.content)
                for msg in messages
            )

            if not is_retry:
                for callback in agent.config.callbacks:
                    context = callback.before_llm_call(context, messages, **kwargs)

            output = self._original_llm_call(messages, **kwargs)

            if not is_retry:
                for callback in agent.config.callbacks:
                    context = callback.after_llm_call(context, output)

            context.framework_state._message_getter = None
            context.framework_state._message_setter = None

            return output

        agent._agent.model.generate = wrap_generate

        def wrapped_tool_execution(original_tool, original_call, *args, **kwargs):
            trace_id = get_current_span().get_span_context().trace_id

            if trace_id == 0 or trace_id not in self.callback_context:
                return original_call(**kwargs)

            context = self.callback_context[trace_id]
            context.shared["original_tool"] = original_tool

            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context, *args, **kwargs)

            output = original_call(**kwargs)

            for callback in agent.config.callbacks:
                context = callback.after_tool_execution(
                    context, output, *args, **kwargs
                )

            return output

        class WrappedToolCall:
            def __init__(self, original_tool, original_forward):
                self.original_tool = original_tool
                self.original_forward = original_forward

            def forward(self, *args, **kwargs):
                return wrapped_tool_execution(
                    self.original_tool, self.original_forward, *args, **kwargs
                )

        self._original_tools = deepcopy(agent._agent.tools)
        wrapped_tools = {}
        for key, tool in agent._agent.tools.items():
            original_forward = tool.forward
            wrapped = WrappedToolCall(tool, original_forward)
            tool.forward = wrapped.forward
            wrapped_tools[key] = tool
        agent._agent.tools = wrapped_tools

    async def unwrap(self, agent: SmolagentsAgent) -> None:
        if self._original_llm_call is not None:
            agent._agent.model.generate = self._original_llm_call
        if self._original_tools is not None:
            agent._agent.tools = self._original_tools
