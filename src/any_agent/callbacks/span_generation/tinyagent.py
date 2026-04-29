"""Re-export tinyagent's span generation callback so identity-based checks line up.

The class lives in `tinyagent.callbacks.span_generation`; importing it through
this module preserves the historic `any_agent.callbacks.span_generation.tinyagent._TinyAgentSpanGeneration`
import path that older code may rely on.
"""

from tinyagent.callbacks.span_generation import (
    _SpanGeneration as _TinyAgentSpanGeneration,
)

__all__ = ["_TinyAgentSpanGeneration"]
