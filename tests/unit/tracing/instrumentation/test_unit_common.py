import json
from unittest.mock import MagicMock

import pytest

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
def test_set_tool_output(tool_output, expected_output, expected_output_type) -> None:
    span_mock = MagicMock()
    _set_tool_output(tool_output, span_mock)

    span_mock.set_attributes.assert_called_with(
        {"gen_ai.output": expected_output, "gen_ai.output.type": expected_output_type}
    )
