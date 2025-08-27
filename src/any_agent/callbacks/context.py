from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

    from any_agent.tracing.agent_trace import AgentTrace


@dataclass
class Context:
    """Object that will be shared across callbacks.

    Each AnyAgent.run has a separate `Context` available.

    `shared` can be used to store and pass information
    across different callbacks.
    """

    current_span: Span
    """You can use the span in your callbacks to get information consistently across frameworks.

    You can find information about the attributes (available under `current_span.attributes`) in
    [Attributes Reference](./tracing.md#any_agent.tracing.attributes).
    """

    trace: AgentTrace
    tracer: Tracer

    shared: dict[str, Any]
    """Can be used to store arbitrary information for sharing across callbacks."""
