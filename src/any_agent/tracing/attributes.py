"""Tracing attribute keys.

`GenAI` is re-exported from `tinyagent` (it tracks the OTel GenAI semantic
conventions). `AnyAgentAttributes` keeps its legacy `any_agent.version` key for
backwards compatibility with existing traces and snapshots.
"""

from tinyagent.tracing.attributes import GenAI


class AnyAgentAttributes:
    """Span-attribute keys specific to AnyAgent library."""

    VERSION = "any_agent.version"
    """The any-agent library version used in the runtime."""


__all__ = ["AnyAgentAttributes", "GenAI"]
