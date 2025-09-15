# ruff: noqa: E721, UP007, UP045, PT006
# mypy: disable-error-code="type-arg"
from typing import Any, Optional, Union

import pytest

from any_agent.utils.cast import safe_cast_argument


@pytest.mark.parametrize(
    "value,target_type,expected",
    [
        ("42", int, 42),
        ("3.14", float, 3.14),
        (42, str, "42"),
        ("hello", str, "hello"),
        (None, Optional[int], None),
        ("42", Union[int, str], 42),
        ("hello", Union[int, str], "hello"),
        ("invalid", int, "invalid"),
        # Empty string to None conversion tests
        ("", str | None, None),
        ("", int | None, None),
        ("", Optional[str], None),
        ("", Union[str, None], None),
        ("", Union[int, str, None], None),
        # Empty string should remain empty for non-optional types
        ("", str, ""),
        ("", int | str, ""),
        ("", Union[int, str], ""),
        # JSON parsing tests - lists
        ('["a", "b"]', list, ["a", "b"]),
        ('["a", "b", "c"]', list, ["a", "b", "c"]),
        (
            '[{"insertText": {"location": {"index": 1}, "text": "Hello"}}]',
            list,
            [{"insertText": {"location": {"index": 1}, "text": "Hello"}}],
        ),
        ("[]", list, []),
        ("[1, 2, 3]", list[int], [1, 2, 3]),
        ('["a", "b", "c"]', list[str], ["a", "b", "c"]),
        ('[{"id": 1}, {"id": 2}]', list[dict], [{"id": 1}, {"id": 2}]),
        # JSON parsing tests - dicts
        ('{"key": "value"}', dict, {"key": "value"}),
        ('{"key": "value", "number": 42}', dict, {"key": "value", "number": 42}),
        ('{"outer": {"inner": "value"}}', dict, {"outer": {"inner": "value"}}),
        ("{}", dict, {}),
        ('{"a": 1, "b": 2}', dict[str, int], {"a": 1, "b": 2}),
        (
            '{"key1": "value1", "key2": "value2"}',
            dict[str, str],
            {"key1": "value1", "key2": "value2"},
        ),
        (
            '{"string": "value", "number": 42, "bool": true}',
            dict[str, Any],
            {"string": "value", "number": 42, "bool": True},
        ),
        (
            '{"a": 1, "b": 2, "c": "three"}',
            dict[str, int | str],
            {"a": 1, "b": 2, "c": "three"},
        ),
        # Invalid JSON should return original
        ('["invalid json"', list, '["invalid json"'),
        ('{"invalid": json}', dict, '{"invalid": json}'),
        ("not json", list, "not json"),
        ("not json", dict, "not json"),
        # Type mismatches should return original
        ('["not", "a", "dict"]', dict, '["not", "a", "dict"]'),
        ('{"not": "a list"}', list, '{"not": "a list"}'),
        ('["list"]', dict, '["list"]'),
        ('{"dict": "value"}', list, '{"dict": "value"}'),
        # Union with complex types
        ('["a", "b"]', list | str, ["a", "b"]),
        ('"plain string"', list | str, '"plain string"'),
        ("plain string", list | str, "plain string"),
        ('{"key": "value"}', dict | str, {"key": "value"}),
        ("plain string", dict | str, "plain string"),
        ('["item"]', list | None, ["item"]),
        # Whitespace handling
        ('  ["a", "b", "c"]  ', list, ["a", "b", "c"]),
        ("", list, ""),
        ("   ", list, "   "),
        # Already correct types
        (["a", "b", "c"], list, ["a", "b", "c"]),
        ({"key": "value"}, dict, {"key": "value"}),
        # Complex data types with None as valid option
        ('["a", "b", "c"]', list[str] | None, ["a", "b", "c"]),
        (None, list[str] | None, None),
        ("", list[str] | None, None),
        ("invalid json", list[str] | None, "invalid json"),
        ('{"key": "value"}', dict[str, str] | None, {"key": "value"}),
        (None, dict[str, str] | None, None),
        ("", dict[str, str] | None, None),
        ("invalid json", dict[str, str] | None, "invalid json"),
        ("[1, 2, 3]", list[int] | None, [1, 2, 3]),
        (None, list[int] | None, None),
        ("", list[int] | None, None),
        ('{"a": 1, "b": 2}', dict[str, int] | None, {"a": 1, "b": 2}),
        (None, dict[str, int] | None, None),
        ("", dict[str, int] | None, None),
        # Using Union syntax for complex types with None
        ('["x", "y"]', Union[list[str], None], ["x", "y"]),
        (None, Union[list[str], None], None),
        ("", Union[list[str], None], None),
        ('{"id": 42}', Union[dict[str, int], None], {"id": 42}),
        (None, Union[dict[str, int], None], None),
        ("", Union[dict[str, int], None], None),
        # Optional complex types
        ('[{"name": "test"}]', Optional[list[dict]], [{"name": "test"}]),
        (None, Optional[list[dict]], None),
        ("", Optional[list[dict]], None),
        (
            '{"nested": {"value": "data"}}',
            Optional[dict[str, dict]],
            {"nested": {"value": "data"}},
        ),
        (None, Optional[dict[str, dict]], None),
        ("", Optional[dict[str, dict]], None),
        # Multi-type unions with complex types and None
        ('["item1", "item2"]', list[str] | dict[str, str] | None, ["item1", "item2"]),
        ('{"key": "value"}', list[str] | dict[str, str] | None, {"key": "value"}),
        (None, list[str] | dict[str, str] | None, None),
        ("", list[str] | dict[str, str] | None, None),
        ("invalid", list[str] | dict[str, str] | None, "invalid"),
        ("false", bool, False),
        ("true", bool, True),
        ("False", bool, False),
        ("True", bool, True),
        ("0", bool, False),
        ("1", bool, True),
        (0, bool, False),
        (1, bool, True),
        (True, str, "True"),
        (False, str, "False"),
        (None, int, None),
        (None, str, None),
        # Failed casting scenarios
        ("not_a_float", float, "not_a_float"),
        # Union type precedence tests
        ("3.14", float | int | str, 3.14),
        ("42", float | int | str, 42.0),
        ("not_a_number", int | str, "not_a_number"),
        ("invalid_for_both", int | float, "invalid_for_both"),
        # Zero values
        (0, str, "0"),
        ("0", int, 0),
    ],
)
def test_parametrized_casting(value: Any, target_type: Any, expected: Any) -> None:
    """Parametrized test for various casting scenarios."""
    assert safe_cast_argument(value, target_type) == expected


def test_modern_vs_traditional_union_consistency() -> None:
    """Test that modern and traditional union types behave consistently."""
    modern_union = int | str | None
    traditional_union = Union[int, str, None]

    test_values = ["42", "hello", None, "invalid_int"]

    for value in test_values:
        modern_result = safe_cast_argument(value, modern_union)
        traditional_result = safe_cast_argument(value, traditional_union)
        assert modern_result == traditional_result
        assert type(modern_result) == type(traditional_result)
