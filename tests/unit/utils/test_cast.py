# ruff: noqa: E712, E721, UP007, UP045, PT006

from typing import Any, Optional, Union

import pytest

from any_agent.utils.cast import safe_cast_argument


def test_basic_type_casting() -> None:
    """Test basic type casting for primitive types."""
    assert safe_cast_argument("42", int) == 42
    assert safe_cast_argument(42, str) == "42"
    assert safe_cast_argument("3.14", float) == 3.14
    assert safe_cast_argument("true", bool) == True
    assert safe_cast_argument("false", bool) == True  # Any non-empty string is truthy
    assert safe_cast_argument("", bool) == False
    assert safe_cast_argument(0, bool) == False
    assert safe_cast_argument(1, bool) == True


def test_none_value_handling() -> None:
    """Test that None values are handled correctly."""
    assert safe_cast_argument(None, int) is None
    assert safe_cast_argument(None, str) is None
    assert safe_cast_argument(None, Optional[int]) is None


def test_failed_casting_returns_original() -> None:
    """Test that failed casting returns the original value."""
    assert safe_cast_argument("invalid", int) == "invalid"
    assert safe_cast_argument("not_a_float", float) == "not_a_float"
    assert safe_cast_argument(object(), str) != object()  # str() works on objects


def test_modern_union_types() -> None:
    """Test modern union types using | syntax."""
    # Single non-None type in union
    union_type = int | None
    assert safe_cast_argument("42", union_type) == 42
    assert safe_cast_argument(None, union_type) is None

    # Multiple types in union - should try each until one works
    multi_union = int | str | None
    assert safe_cast_argument("42", multi_union) == 42
    assert safe_cast_argument("hello", multi_union) == "hello"
    assert safe_cast_argument(None, multi_union) is None


def test_typing_union_types() -> None:
    """Test traditional typing.Union types."""
    # Single non-None type in union
    union_type = Union[int, None]
    assert safe_cast_argument("42", union_type) == 42
    assert safe_cast_argument(None, union_type) is None

    # Multiple types in union
    multi_union = Union[int, str, None]
    assert safe_cast_argument("42", multi_union) == 42
    assert safe_cast_argument("hello", multi_union) == "hello"
    assert safe_cast_argument(None, multi_union) is None


def test_optional_types() -> None:
    """Test Optional type annotations."""
    assert safe_cast_argument("42", Optional[int]) == 42
    assert safe_cast_argument(None, Optional[int]) is None
    assert safe_cast_argument("invalid", Optional[int]) == "invalid"


def test_union_type_precedence() -> None:
    """Test that union types try casting in order."""
    # int should be tried before str, so "42" becomes 42
    union_type = int | str
    assert safe_cast_argument("42", union_type) == 42
    assert isinstance(safe_cast_argument("42", union_type), int)

    # If int casting fails, should fall back to str
    assert safe_cast_argument("hello", union_type) == "hello"
    assert isinstance(safe_cast_argument("hello", union_type), str)


def test_complex_union_scenarios() -> None:
    """Test complex union type scenarios."""
    # Union with float and int - should prefer the first one that works
    union_type = float | int | str
    assert safe_cast_argument("3.14", union_type) == 3.14
    assert isinstance(safe_cast_argument("3.14", union_type), float)

    assert safe_cast_argument("42", union_type) == 42.0  # float("42") works first
    assert isinstance(safe_cast_argument("42", union_type), float)


def test_edge_cases() -> None:
    """Test edge cases and boundary conditions."""
    # Empty string
    assert safe_cast_argument("", str) == ""
    assert safe_cast_argument("", int) == ""  # Casting fails, returns original

    # Zero values
    assert safe_cast_argument(0, str) == "0"
    assert safe_cast_argument("0", int) == 0

    # Boolean edge cases
    assert safe_cast_argument(True, str) == "True"
    assert safe_cast_argument(False, str) == "False"
    assert safe_cast_argument("True", bool) == True
    assert safe_cast_argument("False", bool) == True  # Non-empty string is truthy


def test_empty_string_to_none_conversion() -> None:
    """Test that empty strings are converted to None for optional types."""
    assert safe_cast_argument("", str | None) is None

    assert safe_cast_argument("", Union[str, None]) is None
    assert safe_cast_argument("", Union[int, None]) is None
    assert safe_cast_argument("", Optional[float]) is None
    assert safe_cast_argument("", int | str | None) is None
    assert safe_cast_argument("", Union[int, str, None]) is None


def test_type_already_correct() -> None:
    """Test when the value is already the correct type."""
    assert safe_cast_argument(42, int) == 42
    assert safe_cast_argument("hello", str) == "hello"
    assert safe_cast_argument(3.14, float) == 3.14
    assert safe_cast_argument(True, bool) == True


def test_union_with_failed_casts() -> None:
    """Test union types where some casts fail."""
    # Create a union where int casting fails but str works
    union_type = int | str
    result = safe_cast_argument("not_a_number", union_type)
    assert result == "not_a_number"
    assert isinstance(result, str)


def test_all_union_casts_fail() -> None:
    """Test when all union type casts fail."""
    # This is a bit contrived since str() usually works on most objects
    # But we can test with a custom object that might cause issues
    union_type = int | float
    result = safe_cast_argument("invalid_for_both", union_type)
    assert result == "invalid_for_both"  # Returns original value


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
