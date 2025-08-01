from collections.abc import Sequence
from typing import Any

import pytest
from agents.tool import Tool as OpenaiClass
from google.adk.tools import FunctionTool as GoogleClass
from langchain_core.tools import BaseTool as LangchainClass
from llama_index.core.tools import FunctionTool as LlamaindexClass
from smolagents.tools import Tool as SmolagentsClass

from any_agent import AgentFramework
from any_agent.config import Tool
from any_agent.tools import search_web, visit_webpage
from any_agent.tools.wrappers import _wrap_tools
from any_agent.utils import run_async_in_sync


def wrap_sync(
    tools: Sequence[Tool],
    framework: AgentFramework,
) -> list[Tool]:
    wrapped_tools, _ = run_async_in_sync(_wrap_tools(tools, framework))
    return wrapped_tools


@pytest.mark.parametrize(
    ("framework", "expected_class"),
    [
        (AgentFramework.GOOGLE, GoogleClass),
        (AgentFramework.LANGCHAIN, LangchainClass),
        (AgentFramework.LLAMA_INDEX, LlamaindexClass),
        (AgentFramework.OPENAI, OpenaiClass),
        (AgentFramework.SMOLAGENTS, SmolagentsClass),
    ],
)
def test_wrap_tools(framework: AgentFramework, expected_class: Any) -> None:
    wrapped_tools = wrap_sync([search_web, visit_webpage], framework)
    assert all(isinstance(tool, expected_class) for tool in wrapped_tools)
