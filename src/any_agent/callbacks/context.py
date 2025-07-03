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
    trace: AgentTrace
    tracer: Tracer
    shared: dict[str, Any]
