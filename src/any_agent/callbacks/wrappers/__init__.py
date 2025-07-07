from __future__ import annotations

from typing import assert_never

from any_agent import AgentFramework

from .agno import _AgnoWrapper
from .google import _GoogleADKWrapper
from .langchain import _LangChainWrapper
from .llama_index import _LlamaIndexWrapper
from .openai import _OpenAIAgentsWrapper
from .smolagents import _SmolagentsWrapper
from .tinyagent import _TinyAgentWrapper


def _get_wrapper_by_framework(
    framework: AgentFramework,
) -> (
    _AgnoWrapper
    | _GoogleADKWrapper
    | _LangChainWrapper
    | _LlamaIndexWrapper
    | _OpenAIAgentsWrapper
    | _SmolagentsWrapper
    | _TinyAgentWrapper
):
    if framework is AgentFramework.AGNO:
        return _AgnoWrapper()

    if framework is AgentFramework.GOOGLE:
        return _GoogleADKWrapper()

    if framework is AgentFramework.LANGCHAIN:
        return _LangChainWrapper()

    if framework is AgentFramework.LLAMA_INDEX:
        return _LlamaIndexWrapper()

    if framework is AgentFramework.OPENAI:
        return _OpenAIAgentsWrapper()

    if framework is AgentFramework.SMOLAGENTS:
        return _SmolagentsWrapper()

    if framework is AgentFramework.TINYAGENT:
        return _TinyAgentWrapper()

    assert_never(framework)
