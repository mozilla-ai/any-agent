import asyncio
from unittest.mock import MagicMock, patch

import pytest

from any_agent.config import AgentFramework
from any_agent.tools.wrappers import (
    wrap_tool_google,
    wrap_tool_langchain,
    wrap_tool_llama_index,
    wrap_tool_openai,
    wrap_tool_smolagents,
    wrap_tools,
)


def foo() -> None:
    """Print bar."""


def test_wrap_tool_langchain():
    wrapper = MagicMock()
    with patch("langchain_core.tools.tool", wrapper):
        wrap_tool_langchain(foo)
        wrapper.assert_called_with(foo)


def test_wrap_tool_langchain_already_wrapped():
    from langchain_core.tools import tool

    wrapped = tool(foo)
    wrapper = MagicMock()
    with patch("langchain_core.tools.tool", wrapper):
        wrap_tool_langchain(wrapped)
        wrapper.assert_not_called()


def test_wrap_tool_llama_index():
    from llama_index.core.tools import FunctionTool

    wrapper = MagicMock()
    with patch.object(FunctionTool, "from_defaults", wrapper):
        wrap_tool_llama_index(foo)
        wrapper.assert_called_with(foo)


def test_wrap_tool_llama_index_already_wrapped():
    from llama_index.core.tools import FunctionTool

    wrapped = FunctionTool.from_defaults(foo)
    wrapper = MagicMock()
    with patch.object(FunctionTool, "from_defaults", wrapper):
        wrap_tool_llama_index(wrapped)
        wrapper.assert_not_called()


def test_wrap_tool_openai():
    wrapper = MagicMock()
    with patch("agents.function_tool", wrapper):
        wrap_tool_openai(foo)
        wrapper.assert_called_with(foo)


def test_wrap_tool_openai_already_wrapped():
    from agents import function_tool

    wrapped = function_tool(foo)
    wrapper = MagicMock()
    with patch("agents.function_tool", wrapper):
        wrap_tool_openai(wrapped)
        wrapper.assert_not_called()


def test_wrap_tool_openai_builtin_tools():
    from agents.tool import WebSearchTool

    wrapper = MagicMock()
    with patch("agents.function_tool", wrapper):
        wrap_tool_openai(WebSearchTool())
        wrapper.assert_not_called()


def test_wrap_tool_smolagents():
    wrapper = MagicMock()
    with patch("smolagents.tool", wrapper):
        wrap_tool_smolagents(foo)
        # Check that wrapper was called once with any argument
        wrapper.assert_called_once()
        # The first argument to wrapper should be a function (the wrapped version of foo)
        args, _ = wrapper.call_args
        wrapped_func = args[0]
        # Verify the wrapped function has the same metadata as foo
        assert wrapped_func.__name__ == foo.__name__
        assert wrapped_func.__doc__ == foo.__doc__


def test_wrap_tool_smolagents_already_wrapped():
    from smolagents import tool

    wrapped = tool(foo)
    wrapper = MagicMock()
    with patch("smolagents.tool", wrapper):
        wrap_tool_smolagents(wrapped)
        wrapper.assert_not_called()


def test_wrap_tool_smolagents_builtin_tools():
    from smolagents import DuckDuckGoSearchTool

    wrapper = MagicMock()
    with patch("smolagents.tool", wrapper):
        wrap_tool_smolagents(DuckDuckGoSearchTool())
        wrapper.assert_not_called()


def test_wrap_tool_google():
    from google.adk.tools import FunctionTool

    wrapper = MagicMock()
    wrapper.return_value = None

    with patch.object(FunctionTool, "__init__", wrapper):
        wrap_tool_google(foo)
        wrapper.assert_called_with(foo)


def test_wrap_tool_google_already_wrapped():
    from google.adk.tools import FunctionTool

    wrapper = MagicMock()
    wrapper.return_value = None
    wrapped = FunctionTool(foo)

    with patch.object(FunctionTool, "__init__", wrapper):
        wrap_tool_google(wrapped)
        wrapper.assert_not_called()


def test_wrap_tool_google_builtin_tools():
    from google.adk.tools import FunctionTool, google_search

    wrapper = MagicMock()
    wrapper.return_value = None

    with patch.object(FunctionTool, "__init__", wrapper):
        wrap_tool_google(google_search)
        wrapper.assert_not_called()


frameworks = list(AgentFramework)


@pytest.mark.parametrize(
    "framework",
    frameworks,
)
def test_bad_functions(framework):
    """Test the verify_callable function with various bad functions."""

    # Test missing return type
    def missing_return_type(foo: str):
        """Docstring for foo."""
        return foo

    with pytest.raises(ValueError, match="return type"):
        asyncio.run(wrap_tools([missing_return_type], framework))

    # Test missing docstring
    def missing_docstring(foo: str) -> str:
        return foo

    with pytest.raises(ValueError, match="docstring"):
        asyncio.run(wrap_tools([missing_docstring], framework))

    # Test missing parameter type
    def missing_param_type(foo) -> str:
        """Docstring for foo."""
        return foo

    with pytest.raises(ValueError, match="typed arguments"):
        asyncio.run(wrap_tools([missing_param_type], framework))

    # Good function should not raise an error
    def good_function(foo: str) -> str:
        """Docstring for foo.
        Args:
            foo: The foo argument.
        Returns:
            The foo result.
        """
        return foo

    asyncio.run(wrap_tools([good_function], framework))
