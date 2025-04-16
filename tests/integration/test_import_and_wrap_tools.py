import asyncio

import pytest
from agents.tool import Tool as OpenaiClass
from google.adk.tools import FunctionTool as GoogleClass
from langchain.tools import BaseTool as LangchainClass
from llama_index.core.tools import FunctionTool as LlamaindexClass
from smolagents.tools import Tool as SmolagentsClass

from any_agent import AgentFramework
from any_agent.tools import search_web, visit_webpage
from any_agent.tools.wrappers import wrap_tools


def wrap_sync(tools, framework):
    wrapped_tools, _ = asyncio.get_event_loop().run_until_complete(
        wrap_tools(tools, framework)
    )
    return wrapped_tools


@pytest.mark.parametrize(
    ("framework", "expected_class"),
    [
        ("google", GoogleClass),
        ("langchain", LangchainClass),
        ("llama_index", LlamaindexClass),
        ("openai", OpenaiClass),
        ("smolagents", SmolagentsClass),
    ],
)
def test_wrap_tools(framework, expected_class):
    wrapped_tools = wrap_sync([search_web, visit_webpage], AgentFramework(framework))
    assert all(isinstance(tool, expected_class) for tool in wrapped_tools)
