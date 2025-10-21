"""Vendoring https://github.com/Akshay-Dongare/langchain-litellm/blob/main/langchain_litellm/chat_models/litellm.py and edited to use any-llm instead of LiteLLM. Marking as vendored since the logic contained here is nearly identical to the litellm version."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import (
    Any,
)

from any_llm.types.completion import ChoiceDelta
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    if role == "assistant":
        content = _dict.get("content", "") or ""

        additional_kwargs = {}
        if _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(_dict["function_call"])

        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "system":
        return SystemMessage(content=_dict["content"])
    if role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    if role == "tool":
        return ToolMessage(content=_dict["content"], tool_call_id=_dict["tool_call_id"])
    return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    delta: ChoiceDelta, default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = delta.role
    content = delta.content or ""
    additional_kwargs: dict[str, Any] = {}
    if delta.function_call:
        additional_kwargs["function_call"] = dict(delta.function_call)
    reasoning = getattr(delta, "reasoning", None)
    if reasoning and reasoning.content:
        additional_kwargs["reasoning_content"] = reasoning.content

    tool_call_chunks = []
    if raw_tool_calls := delta.tool_calls:
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                ToolCallChunk(
                    name=rtc.function.name if rtc.function else "",
                    args=rtc.function.arguments if rtc.function else "",
                    id=rtc.id,
                    index=rtc.index,
                )
                for rtc in raw_tool_calls
                if rtc.function
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if default_class == FunctionMessageChunk:
        if delta.function_call:
            return FunctionMessageChunk(
                content=delta.function_call.arguments or "",
                name=delta.function_call.name or "",
            )
    if role == "tool" or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    return default_class(content=content)  # type: ignore[call-arg]


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    message_dict: dict[str, Any] = {"content": message.content}
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
        message_dict["name"] = message.name
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        error_message = f"Got unknown type {message}"
        raise ValueError(error_message)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict
