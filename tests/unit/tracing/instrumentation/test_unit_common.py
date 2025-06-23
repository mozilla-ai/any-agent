import json
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.trace import StatusCode

from any_agent.tracing.instrumentation.common import _set_tool_output


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
    span_mock = MagicMock()
    _set_tool_output(tool_output, span_mock)

    span_mock.set_attributes.assert_called_with(
        {"gen_ai.output": expected_output, "gen_ai.output.type": expected_output_type}
    )
    span_mock.set_status.assert_called_with(StatusCode.OK)


def test_set_tool_output_error() -> None:
    error = "Error calling tool: It's a trap!"
    span_mock = MagicMock()
    status_mock = MagicMock()
    with patch("any_agent.tracing.instrumentation.common.Status", status_mock):
        _set_tool_output(error, span_mock)

        span_mock.set_attributes.assert_called_with(
            {"gen_ai.output": error, "gen_ai.output.type": "text"}
        )
        span_mock.set_status.assert_called_with(
            status_mock(status_code=StatusCode.ERROR, description=error)
        )
