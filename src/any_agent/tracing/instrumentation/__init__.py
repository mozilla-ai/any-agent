from __future__ import annotations

from typing import assert_never

from any_agent import AgentFramework

from .agno import _AgnoInstrumentor
from .google import _GoogleADKInstrumentor
from .langchain import _LangChainInstrumentor
from .llama_index import _LlamaIndexInstrumentor
from .openai import _OpenAIAgentsInstrumentor
from .smolagents import _SmolagentsInstrumentor
from .tinyagent import _TinyAgentInstrumentor


def _get_instrumentor_by_framework(
    framework: AgentFramework,
) -> (
    _AgnoInstrumentor
    | _GoogleADKInstrumentor
    | _LangChainInstrumentor
    | _LlamaIndexInstrumentor
    | _OpenAIAgentsInstrumentor
    | _SmolagentsInstrumentor
    | _TinyAgentInstrumentor
):
    if framework is AgentFramework.AGNO:
        return _AgnoInstrumentor()

    if framework is AgentFramework.GOOGLE:
        return _GoogleADKInstrumentor()

    if framework is AgentFramework.LANGCHAIN:
        return _LangChainInstrumentor()

    if framework is AgentFramework.LLAMA_INDEX:
        return _LlamaIndexInstrumentor()

    if framework is AgentFramework.OPENAI:
        return _OpenAIAgentsInstrumentor()

    if framework is AgentFramework.SMOLAGENTS:
        return _SmolagentsInstrumentor()

    if framework is AgentFramework.TINYAGENT:
        return _TinyAgentInstrumentor()

    assert_never(framework)
