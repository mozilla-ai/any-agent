import types
from typing import Any, Union, get_args, get_origin


def _is_optional_type(arg_type: Any) -> bool:
    """Check if a type is optional (contains None as a union member)."""
    # Handle modern union types (e.g., int | str | None)
    if isinstance(arg_type, types.UnionType):
        union_args = get_args(arg_type)
        return type(None) in union_args

    # Handle typing.Union (older style)
    if get_origin(arg_type) is Union:
        union_args = get_args(arg_type)
        return type(None) in union_args

    return False


def safe_cast_argument(value: Any, arg_type: Any) -> Any:
    """Safely cast an argument to the specified type, handling union types.

    Args:
        value: The value to cast
        arg_type: The target type (may be a union type)

    Returns:
        The cast value, or the original value if casting fails

    """
    # Handle None values for optional types
    if value is None:
        return None

    # If you get an empty str and None is an option, return it as None
    if value == "" and _is_optional_type(arg_type):
        return None

    # Handle modern union types (e.g., int | str | None)
    if isinstance(arg_type, types.UnionType):
        union_args = get_args(arg_type)
        # Filter out NoneType for optional parameters
        non_none_types = [t for t in union_args if t is not type(None)]

        if len(non_none_types) == 1:
            try:
                return non_none_types[0](value)
            except (ValueError, TypeError):
                return value

        # For multiple types, try each one until one works
        for cast_type in non_none_types:
            try:
                return cast_type(value)
            except (ValueError, TypeError):
                continue
        return value

    # Handle typing.Union (older style)
    if get_origin(arg_type) is Union:
        union_args = get_args(arg_type)
        # Filter out NoneType for optional parameters
        non_none_types = [t for t in union_args if t is not type(None)]

        # If only one non-None type, try to cast to it
        if len(non_none_types) == 1:
            try:
                return non_none_types[0](value)
            except (ValueError, TypeError):
                return value

        # For multiple types, try each one until one works
        for cast_type in non_none_types:
            try:
                return cast_type(value)
            except (ValueError, TypeError):
                continue
        return value

    try:
        return arg_type(value)
    except (ValueError, TypeError):
        return value
