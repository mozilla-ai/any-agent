"""Round-trip tests for the vendored any-llm/LangChain message converters.

Issue #967: thinking models (Kimi, DeepSeek) reject the second request of a
tool loop when `reasoning_content` is dropped from the assistant turn, and
any-llm's `extra_content` (DeepSeek reasoning, Anthropic signatures) must
survive the LangChain round trip as well.
"""

from typing import Any

from any_agent.vendor.langchain_any_llm import (
    _convert_dict_to_message,
    _convert_message_to_dict,
)

PROVIDER_ASSISTANT_DICT: dict[str, Any] = {
    "content": "",
    "role": "assistant",
    "reasoning_content": "the user asks about weather; call the tool",
    "extra_content": {"anthropic": {"signature": "sig-1"}},
    "tool_calls": [
        {
            "type": "function",
            "id": "call_1",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            "extra_content": {"anthropic": {"signature": "sig-tool-1"}},
        }
    ],
}


def test_reasoning_content_survives_the_round_trip() -> None:
    message = _convert_dict_to_message(PROVIDER_ASSISTANT_DICT)
    assert message.additional_kwargs["reasoning_content"] == (
        "the user asks about weather; call the tool"
    )
    wire = _convert_message_to_dict(message)
    assert wire["reasoning_content"] == "the user asks about weather; call the tool"


def test_message_and_tool_call_extra_content_survive_the_round_trip() -> None:
    message = _convert_dict_to_message(PROVIDER_ASSISTANT_DICT)
    wire = _convert_message_to_dict(message)
    assert wire["extra_content"] == {"anthropic": {"signature": "sig-1"}}
    assert wire["tool_calls"][0]["extra_content"] == {
        "anthropic": {"signature": "sig-tool-1"}
    }


def test_tool_call_shape_is_unchanged_without_extras() -> None:
    plain = {
        "content": "",
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }
    message = _convert_dict_to_message(plain)
    wire = _convert_message_to_dict(message)
    assert wire["tool_calls"] == [
        {
            "type": "function",
            "id": "call_1",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ]
    assert "reasoning_content" not in wire
    assert "extra_content" not in wire
