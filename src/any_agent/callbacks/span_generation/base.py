"""Re-export the _SpanGeneration helper base class from `tinyagent`.

The framework-specific `_*SpanGeneration` callbacks in this package extend this
base for its `_set_llm_input` / `_set_llm_output` / `_set_tool_input` /
`_set_tool_output` helpers; aliasing the class here means `isinstance` checks
across any-agent and tinyagent agree.
"""

from tinyagent.callbacks.span_generation import _SpanGeneration

__all__ = ["_SpanGeneration"]
