import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)

from any_agent.callbacks.span_generation.openai import _OpenAIAgentsSpanGeneration
from any_agent.tracing.attributes import GenAI


def _tool_call(name: str, arguments: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call", call_id=f"call_{name}", name=name, arguments=arguments
    )


def _reasoning_item() -> ResponseReasoningItem:
    return ResponseReasoningItem(id="rs_1", type="reasoning", summary=[])


def _text_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _run_after_llm_call(output: list[object]) -> dict[str, object]:
    context = MagicMock()
    response = SimpleNamespace(
        output=output, usage=SimpleNamespace(input_tokens=10, output_tokens=5)
    )
    _OpenAIAgentsSpanGeneration().after_llm_call(context, response)
    attributes: dict[str, object] = context.current_span.set_attributes.call_args[0][0]
    return attributes


def test_after_llm_call_records_all_parallel_tool_calls() -> None:
    attributes = _run_after_llm_call(
        [
            _tool_call("get_weather", '{"city": "Paris"}'),
            _tool_call("get_time", '{"tz": "CET"}'),
        ]
    )

    assert json.loads(str(attributes[GenAI.OUTPUT])) == [
        {"tool.name": "get_weather", "tool.args": '{"city": "Paris"}'},
        {"tool.name": "get_time", "tool.args": '{"tz": "CET"}'},
    ]
    assert attributes[GenAI.USAGE_INPUT_TOKENS] == 10
    assert attributes[GenAI.USAGE_OUTPUT_TOKENS] == 5


def test_after_llm_call_records_tool_call_after_text_and_reasoning() -> None:
    attributes = _run_after_llm_call(
        [
            _reasoning_item(),
            _text_message("Let me check."),
            _tool_call("get_weather", '{"city": "Paris"}'),
        ]
    )

    assert json.loads(str(attributes[GenAI.OUTPUT])) == [
        {"tool.name": "get_weather", "tool.args": '{"city": "Paris"}'}
    ]


def test_after_llm_call_records_text_after_reasoning() -> None:
    attributes = _run_after_llm_call([_reasoning_item(), _text_message("Sunny.")])

    assert attributes[GenAI.OUTPUT] == "Sunny."
    assert attributes[GenAI.OUTPUT_TYPE] == "text"
