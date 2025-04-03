from unittest.mock import MagicMock, patch

from any_agent.tools.wrappers import (
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
        wrap_tool_langchain("any_agent.tools.search_web")
        wrapper.assert_called_with("any_agent.tools.search_web")


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
        wrap_tool_llama_index("any_agent.tools.search_web")
        wrapper.assert_called_with("any_agent.tools.search_web")


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
        wrap_tool_openai("any_agent.tools.search_web")
        wrapper.assert_called_with("any_agent.tools.search_web")


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
        wrap_tool_smolagents("any_agent.tools.search_web")
        wrapper.assert_called_with("any_agent.tools.search_web")


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
