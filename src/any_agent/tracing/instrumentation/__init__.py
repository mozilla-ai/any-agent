from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, assert_never

from any_agent import AgentFramework

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


class _Instrumentor(Protocol):
    def instrument(self, *, tracer: Tracer) -> None: ...

    def uninstrument(self) -> None: ...


def _get_instrumentor_by_framework(framework: AgentFramework) -> _Instrumentor:
    if framework is AgentFramework.AGNO:
        from .agno import _AgnoInstrumentor

        return _AgnoInstrumentor()

    if framework is AgentFramework.GOOGLE:
        from .google import _GoogleADKInstrumentor

        return _GoogleADKInstrumentor()

    if framework is AgentFramework.LANGCHAIN:
        from .langchain import _LangChainInstrumentor

        return _LangChainInstrumentor()

    if framework is AgentFramework.LLAMA_INDEX:
        from .llama_index import _LlamaIndexInstrumentor

        return _LlamaIndexInstrumentor()

    if framework is AgentFramework.OPENAI:
        from .openai import _OpenAIAgentsInstrumentor

        return _OpenAIAgentsInstrumentor()

    if framework is AgentFramework.SMOLAGENTS:
        from .smolagents import _SmolagentsInstrumentor

        return _SmolagentsInstrumentor()

    if framework is AgentFramework.TINYAGENT:
        from .tinyagent import _TinyAgentInstrumentor

        return _TinyAgentInstrumentor()

    assert_never(framework)
