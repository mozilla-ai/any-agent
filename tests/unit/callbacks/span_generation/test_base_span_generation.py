import json
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.trace import StatusCode

from any_agent.callbacks.span_generation.base import _SpanGeneration
from any_agent.tracing.attributes import GenAI


class FooClass:
    pass


foo_instance = FooClass()


@pytest.mark.parametrize(
    ("tool_output", "expected_output", "expected_output_type"),
    [
        ("foo", "foo", "text"),
        (json.dumps({"foo": "bar"}), json.dumps({"foo": "bar"}), "json"),
        ({"foo": "bar"}, json.dumps({"foo": "bar"}), "json"),
        (foo_instance, json.dumps(foo_instance, default=str), "json"),
    ],
)
def test_set_tool_output(
    tool_output: object, expected_output: str, expected_output_type: str
) -> None:
    context = MagicMock()
    _SpanGeneration()._set_tool_output(context, tool_output)

    context.current_span.set_attributes.assert_called_with(
        {
            GenAI.OUTPUT: expected_output,
            GenAI.OUTPUT_TYPE: expected_output_type,
        }
    )
    context.current_span.set_status.assert_called_with(StatusCode.OK)


def test_set_tool_output_error() -> None:
    error = "Error calling tool: It's a trap!"
    context = MagicMock()
    status_mock = MagicMock()
    with patch("any_agent.callbacks.span_generation.base.Status", status_mock):
        _SpanGeneration()._set_tool_output(context, error)

        context.current_span.set_attributes.assert_called_with(
            {GenAI.OUTPUT: error, GenAI.OUTPUT_TYPE: "text"}
        )
        context.current_span.set_status.assert_called_with(
            status_mock(status_code=StatusCode.ERROR, description=error)
        )


def test_set_llm_input() -> None:
    context = MagicMock()

    span_generation = _SpanGeneration()
    span_generation._set_llm_input(context, model_id="gpt-5", input_messages=[])
    context.current_span.set_attribute.assert_called_with(GenAI.INPUT_MESSAGES, "[]")

    # first_llm_call logic should avoid logging input_messages
    # on subsequent calls.
    span_generation._set_llm_input(context, model_id="gpt-5", input_messages=[])
    assert context.current_span.set_attribute.call_count == 1


def test_set_llm_output() -> None:
    context = MagicMock()

    span_generation = _SpanGeneration()
    span_generation._set_llm_output(
        context, output="foo", input_tokens=0, output_tokens=0
    )
    context.current_span.set_attributes.assert_any_call(
        {
            GenAI.OUTPUT: "foo",
            GenAI.OUTPUT_TYPE: "text",
        }
    )

    span_generation._set_llm_output(context, output=[], input_tokens=0, output_tokens=0)
    context.current_span.set_attributes.assert_any_call(
        {
            GenAI.OUTPUT: "[]",
            GenAI.OUTPUT_TYPE: "json",
        }
    )


def test_set_tool_input() -> None:
    context = MagicMock()

    span_generation = _SpanGeneration()
    span_generation._set_tool_input(context, name="foo", args={})
    context.current_span.set_attribute.assert_called_with(GenAI.TOOL_ARGS, "{}")
