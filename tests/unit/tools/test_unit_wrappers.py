from unittest.mock import MagicMock, patch

from any_agent.tools.wrappers import (
    wrap_tool_google,
    wrap_tool_langchain,
    wrap_tool_llama_index,
    wrap_tool_openai,
    wrap_tool_smolagents,
)


def foo() -> None:
    """Print bar."""
    print("bar")


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
        wrapper.assert_called_with(foo)


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
