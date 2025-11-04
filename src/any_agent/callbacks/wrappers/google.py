# mypy: disable-error-code="no-untyped-def,union-attr"
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

if TYPE_CHECKING:
    from any_agent.callbacks.context import Context
    from any_agent.frameworks.google import GoogleAgent


def _import_google_converters() -> tuple[Any, Any]:
    """Import conversion functions from google framework module."""
    from any_agent.frameworks.google import (
        _messages_from_content,
        _messages_to_contents,
    )

    return _messages_from_content, _messages_to_contents


def _llm_response_to_message(llm_response) -> dict[str, Any]:
    """Convert Google ADK LlmResponse to a normalized message dict."""
    from any_agent.frameworks.google import ADK_TO_ANY_LLM_ROLE, _safe_json_serialize

    if not llm_response or not llm_response.content:
        return {"role": "assistant", "content": None}

    content = llm_response.content
    role = ADK_TO_ANY_LLM_ROLE.get(str(content.role), "assistant")

    message_content: list[Any] = []
    tool_calls: list[Any] = []

    if content.parts:
        for part in content.parts:
            if part.text:
                message_content.append({"type": "text", "text": part.text})
            elif part.function_call:
                tool_calls.append(
                    {
                        "type": "function",
                        "id": part.function_call.id,
                        "function": {
                            "name": part.function_call.name,
                            "arguments": _safe_json_serialize(part.function_call.args),
                        },
                    }
                )

    return {
        "role": role,
        "content": message_content or None,
        "tool_calls": tool_calls or None,
    }


def _message_to_llm_response(message: dict[str, Any]):
    """Convert a normalized message dict back to Google ADK LlmResponse."""
    import json

    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    parts = []

    content = message.get("content")
    if content:
        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(types.Part.from_text(text=part.get("text", "")))

    tool_calls = message.get("tool_calls")
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                function_data = tool_call.get("function", {})
                args_str = function_data.get("arguments", "{}")
                try:
                    args = (
                        json.loads(args_str) if isinstance(args_str, str) else args_str
                    )
                except json.JSONDecodeError:
                    args = {}

                part = types.Part.from_function_call(
                    name=function_data.get("name", ""),
                    args=args,
                )
                part.function_call.id = tool_call.get("id", "")
                parts.append(part)

    return LlmResponse(content=types.Content(role="model", parts=parts), partial=False)


class _GoogleADKWrapper:
    def __init__(self) -> None:
        self.callback_context: dict[int, Context] = {}
        self._original: dict[str, Any] = {}

    async def wrap(self, agent: GoogleAgent) -> None:
        self._original["before_model"] = agent._agent.before_model_callback

        def before_model_callback(*args, **kwargs) -> Any | None:
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            llm_request = kwargs.get("llm_request")
            if llm_request is not None:
                _messages_from_content, _messages_to_contents = (
                    _import_google_converters()
                )

                messages = _messages_from_content(llm_request)

                for message in messages:
                    content = message.get("content")
                    if isinstance(content, list):
                        text_parts = [
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        ]
                        if text_parts and len(text_parts) == len(content):
                            message["content"] = (
                                " ".join(text_parts)
                                if len(text_parts) > 1
                                else text_parts[0]
                            )

                if llm_request.config and llm_request.config.system_instruction:
                    messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": llm_request.config.system_instruction,
                        },
                    )

                context.framework_state.messages = messages

                def get_messages():
                    return context.framework_state.messages

                def set_messages(new_messages):
                    context.framework_state.messages = new_messages
                    system_instruction, contents = _messages_to_contents(new_messages)
                    llm_request.contents = contents
                    if llm_request.config:
                        llm_request.config.system_instruction = system_instruction

                context.framework_state._message_getter = get_messages
                context.framework_state._message_setter = set_messages

            for callback in agent.config.callbacks:
                context = callback.before_llm_call(context, *args, **kwargs)

            if callable(self._original["before_model"]):
                return self._original["before_model"](*args, **kwargs)

            return None

        agent._agent.before_model_callback = before_model_callback

        self._original["after_model"] = agent._agent.after_model_callback

        def after_model_callback(*args, **kwargs) -> Any | None:
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            llm_response = kwargs.get("llm_response")
            if llm_response is not None:
                response_message = _llm_response_to_message(llm_response)
                existing_messages = context.framework_state.messages.copy()
                if isinstance(response_message.get("content"), list):
                    text_parts = [
                        part.get("text", "")
                        for part in response_message["content"]
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    if text_parts:
                        response_message["content"] = " ".join(text_parts)

                all_messages = [*existing_messages, response_message]
                context.framework_state.messages = all_messages

                original_response_content = response_message.get("content")

                def get_messages():
                    return context.framework_state.messages

                def set_messages(new_messages):
                    context.framework_state.messages = new_messages

                context.framework_state._message_getter = get_messages
                context.framework_state._message_setter = set_messages

            for callback in agent.config.callbacks:
                context = callback.after_llm_call(context, *args, **kwargs)

            context.framework_state._message_getter = None
            context.framework_state._message_setter = None

            if llm_response is not None:
                final_messages = context.framework_state.messages
                if final_messages:
                    final_response_message = final_messages[-1]
                    if (
                        final_response_message.get("content")
                        != original_response_content
                    ):
                        return _message_to_llm_response(final_response_message)

            if callable(self._original["after_model"]):
                return self._original["after_model"](*args, **kwargs)

            return None

        agent._agent.after_model_callback = after_model_callback

        self._original["before_tool"] = agent._agent.before_tool_callback

        def before_tool_callback(*args, **kwargs) -> Any | None:
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context, *args, **kwargs)

            if callable(self._original["before_tool"]):
                return self._original["before_tool"](*args, **kwargs)

            return None

        agent._agent.before_tool_callback = before_tool_callback

        self._original["after_tool"] = agent._agent.after_tool_callback

        def after_tool_callback(*args, **kwarg) -> Any | None:
            context = self.callback_context[
                get_current_span().get_span_context().trace_id
            ]

            for callback in agent.config.callbacks:
                context = callback.after_tool_execution(context, *args, **kwarg)

            if callable(self._original["after_tool"]):
                return self._original["after_tool"](*args, **kwarg)

            return None

        agent._agent.after_tool_callback = after_tool_callback

    async def unwrap(self, agent: GoogleAgent) -> None:
        if "before_model" in self._original:
            agent._agent.before_model_callback = self._original["before_model"]
        if "before_tool" in self._original:
            agent._agent.before_tool_callback = self._original["before_tool"]
        if "after_model" in self._original:
            agent._agent.after_model_callback = self._original["after_model"]
        if "after_tool" in self._original:
            agent._agent.after_tool_callback = self._original["after_tool"]
