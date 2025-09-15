import json
import types
from typing import Any, Union, get_args, get_origin

from any_agent.logging import logger


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

    # Handle JSON string parsing for complex types
    if isinstance(value, str) and value.strip():
        # Try to parse JSON strings for list and dict types
        if arg_type in (list, dict) or (
            hasattr(arg_type, "__origin__") and arg_type.__origin__ in (list, dict)
        ):
            try:
                parsed = json.loads(value)
                if arg_type is list and isinstance(parsed, list):
                    return parsed
                if arg_type is dict and isinstance(parsed, dict):
                    return parsed
                if hasattr(arg_type, "__origin__"):
                    if arg_type.__origin__ is list and isinstance(parsed, list):
                        return parsed
                    if arg_type.__origin__ is dict and isinstance(parsed, dict):
                        return parsed
            except (json.JSONDecodeError, TypeError):
                pass

    # Handle modern union types (e.g., int | str | None)
    if isinstance(arg_type, types.UnionType):
        union_args = get_args(arg_type)
        # Filter out NoneType for optional parameters
        non_none_types = [t for t in union_args if t is not type(None)]

        if len(non_none_types) == 1:
            # Recursively try to cast to the single non-None type
            return safe_cast_argument(value, non_none_types[0])

        # For multiple types, try each one until one works
        for cast_type in non_none_types:
            try:
                result = safe_cast_argument(value, cast_type)
                if result != value:  # If casting actually changed the value, use it
                    return result
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
            # Recursively try to cast to the single non-None type
            return safe_cast_argument(value, non_none_types[0])

        # For multiple types, try each one until one works
        for cast_type in non_none_types:
            try:
                result = safe_cast_argument(value, cast_type)
                if result != value:  # If casting actually changed the value, use it
                    return result
            except (ValueError, TypeError):
                continue
        return value

    if arg_type is bool and isinstance(value, str):
        lower_value = value.lower().strip()
        if lower_value in ("true", "1", "yes"):
            return True
        if lower_value in ("false", "0"):
            return False
        logger.warning("Unexpected boolean string value: %s", value)
        return bool(value)

    # Handle direct type casting for simple types
    if arg_type in (list, dict):
        # If we got here, it means JSON parsing failed above, so return as-is
        logger.warning("Failed to parse JSON string for type %s: %s", arg_type, value)
        return value

    # Handle parameterized generic types (e.g., list[str], dict[str, int])
    if hasattr(arg_type, "__origin__") and arg_type.__origin__ in (list, dict):
        # If we got here, it means JSON parsing failed above, so return as-is
        logger.warning("Failed to parse JSON string for type %s: %s", arg_type, value)
        return value

    try:
        return arg_type(value)
    except (ValueError, TypeError):
        logger.warning("Failed to cast value %s to type %s", value, arg_type)
        return value
