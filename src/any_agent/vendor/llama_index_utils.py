# The following utility functions are directly copied from the llama_index.llms.litellm module.utils module.
# They don't contain any litellm specific code, but in order to import that module from llama_index you would need
# to have litellm installed (it's imported at the top of the file), so we copy the code here to avoid the dependency.
from collections.abc import Sequence
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.base.llms.types import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    TextBlock,
)


def to_openailike_message_dict(message: ChatMessage) -> dict[str, Any]:
    """Convert a ChatMessage to an OpenAI-like message dict."""
    content = []
    content_txt = ""
    for block in message.blocks:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
            content_txt += block.text
        elif isinstance(block, ImageBlock):
            if block.url:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": cast(
                            "Any",
                            {
                                "url": str(block.url),
                                "detail": block.detail or "auto",
                            },
                        ),
                    }
                )
            else:
                img_bytes = block.resolve_image(as_base64=True).read()
                img_str = img_bytes.decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": cast(
                            "Any",
                            {
                                "url": f"data:{block.image_mimetype};base64,{img_str}",
                                "detail": block.detail or "auto",
                            },
                        ),
                    }
                )
        elif isinstance(block, AudioBlock):
            audio_bytes = block.resolve_audio(as_base64=True).read()
            audio_str = audio_bytes.decode("utf-8")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": cast(
                        "Any",
                        {
                            "data": audio_str,
                            "format": block.format,
                        },
                    ),
                }
            )
        elif isinstance(block, DocumentBlock):
            if not block.data:
                file_buffer = block.resolve_document()
                b64_string = block._get_b64_string(file_buffer)
                mimetype = block.document_mimetype or block._guess_mimetype()
            else:
                b64_string = block.data.decode("utf-8")
                mimetype = block.document_mimetype or block._guess_mimetype()
            content.append(
                {
                    "type": "file",
                    "file": cast(
                        "Any",
                        {
                            "file_data": f"data:{mimetype};base64,{b64_string}",
                        },
                    ),
                }
            )
        else:
            msg = f"Unsupported content block type: {type(block).__name__}"
            raise ValueError(msg)

    message_dict = {
        "role": message.role.value,
        "content": (
            content_txt
            if all(isinstance(block, TextBlock) for block in message.blocks)
            else content
        ),
    }

    message_dict.update(message.additional_kwargs)

    return message_dict


def to_openai_message_dicts(
    messages: Sequence[ChatMessage],
) -> list[dict[str, Any]]:
    """Convert generic messages to OpenAI message dicts."""
    return [to_openailike_message_dict(message) for message in messages]


def from_openai_message_dict(message_dict: dict[str, Any]) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = message_dict["role"]
    content = message_dict.get("content")

    additional_kwargs = message_dict.copy()
    additional_kwargs.pop("role")
    additional_kwargs.pop("content", None)

    return ChatMessage(role=role, content=content, additional_kwargs=additional_kwargs)


def update_tool_calls(
    tool_calls: list[dict[str, Any]],
    tool_call_deltas: Any,
) -> list[dict[str, Any]]:
    """Update the list of tool calls with deltas.

    Args:
        tool_calls: The current list of tool calls
        tool_call_deltas: A list of deltas to update tool_calls with

    Returns:
        The updated tool calls

    """
    if not tool_call_deltas:
        return tool_calls

    for tool_call_delta in tool_call_deltas:
        delta_dict: dict[str, Any] = {}
        if hasattr(tool_call_delta, "id") and tool_call_delta.id is not None:
            delta_dict["id"] = tool_call_delta.id
        if hasattr(tool_call_delta, "type") and tool_call_delta.type is not None:
            delta_dict["type"] = tool_call_delta.type
        if hasattr(tool_call_delta, "index"):
            delta_dict["index"] = tool_call_delta.index

        if (
            hasattr(tool_call_delta, "function")
            and tool_call_delta.function is not None
        ):
            delta_dict["function"] = {}
            if (
                hasattr(tool_call_delta.function, "name")
                and tool_call_delta.function.name is not None
            ):
                delta_dict["function"]["name"] = tool_call_delta.function.name
            if (
                hasattr(tool_call_delta.function, "arguments")
                and tool_call_delta.function.arguments is not None
            ):
                delta_dict["function"]["arguments"] = tool_call_delta.function.arguments

        if len(tool_calls) == 0:
            tool_calls.append(delta_dict)
        else:
            found_match = False
            for existing_tool in tool_calls:
                index_match = False
                if "index" in delta_dict and "index" in existing_tool:
                    index_match = delta_dict["index"] == existing_tool["index"]

                id_match = False
                if "id" in delta_dict and "id" in existing_tool:
                    id_match = delta_dict["id"] == existing_tool["id"]

                if index_match or id_match:
                    found_match = True
                    if "function" in delta_dict:
                        if "function" not in existing_tool:
                            existing_tool["function"] = {}

                        if "name" in delta_dict["function"]:
                            if "name" not in existing_tool["function"]:
                                existing_tool["function"]["name"] = ""
                            existing_tool["function"]["name"] += delta_dict[
                                "function"
                            ].get("name", "")

                        if "arguments" in delta_dict["function"]:
                            if "arguments" not in existing_tool["function"]:
                                existing_tool["function"]["arguments"] = ""
                            existing_tool["function"]["arguments"] += delta_dict[
                                "function"
                            ].get("arguments", "")

                    if "id" in delta_dict:
                        if "id" not in existing_tool:
                            existing_tool["id"] = ""
                        existing_tool["id"] += delta_dict.get("id", "")

                    if "type" in delta_dict:
                        existing_tool["type"] = delta_dict["type"]

                    if "index" in delta_dict:
                        existing_tool["index"] = delta_dict["index"]

                    break

            if not found_match and ("id" in delta_dict or "index" in delta_dict):
                tool_calls.append(delta_dict)

    return tool_calls


def force_single_tool_call(response: ChatResponse) -> None:
    """Force a response to have only a single tool call."""
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]
