from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from a2a.types import TaskState  # noqa: TC002
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from any_agent import AnyAgent


class _DefaultBody(BaseModel):
    """Default payload when the user does not supply one."""

    result: str

    model_config = ConfigDict(extra="forbid")


# Define a TypeVar for the body type
BodyType = TypeVar("BodyType", bound=BaseModel)


class A2AEnvelope(BaseModel, Generic[BodyType]):
    """A2A envelope that wraps response data with task status."""

    task_status: TaskState
    data: BodyType

    model_config = ConfigDict(extra="forbid")


def _is_a2a_envelope(typ: type[BaseModel] | None) -> bool:
    if typ is None:
        return False
    fields: Any = getattr(typ, "model_fields", None)

    # We only care about a mapping with the required keys.
    if not isinstance(fields, Mapping):
        return False

    return "task_status" in fields and "data" in fields


def _create_a2a_envelope(body_type: type[BaseModel]) -> type[A2AEnvelope[Any]]:
    """Return a *new* Pydantic model that wraps *body_type* with TaskState + data."""
    # Ensure body forbids extra keys (OpenAI response_format requirement)
    if hasattr(body_type, "model_config"):
        body_type.model_config["extra"] = "forbid"
    else:
        body_type.model_config = ConfigDict(extra="forbid")

    class EnvelopeInstance(A2AEnvelope[body_type]):  # type: ignore[valid-type]
        pass

    EnvelopeInstance.__name__ = f"{body_type.__name__}Return"
    EnvelopeInstance.__qualname__ = f"{body_type.__qualname__}Return"
    return EnvelopeInstance


def prepare_agent_for_a2a(agent: AnyAgent) -> AnyAgent:
    """Return an agent whose ``config.output_type`` is A2A-ready.

    If *agent* is already envelope-compatible we hand it back untouched.
    Otherwise we clone its config, wrap the output type, and spin up a
    *new* agent instance via `AnyAgent.create` so that framework-specific
    initialisation sees the correct schema right from the start.

    This function preserves MCP servers from the original agent to avoid
    connection timeouts.
    """
    if _is_a2a_envelope(agent.config.output_type):
        return agent

    body_type = agent.config.output_type or _DefaultBody
    new_output_type = _create_a2a_envelope(body_type)

    new_config = agent.config.model_copy(deep=True)
    new_config.output_type = new_output_type

    # Create the new agent with the wrapped config, preserving MCP servers and tools
    return agent._recreate_with_config(new_config)


async def prepare_agent_for_a2a_async(agent: AnyAgent) -> AnyAgent:
    """Async counterpart of :pyfunc:`prepare_agent_for_a2a`.

    This function preserves MCP servers from the original agent to avoid
    connection timeouts.
    """
    if _is_a2a_envelope(agent.config.output_type):
        return agent

    body_type = agent.config.output_type or _DefaultBody
    new_output_type = _create_a2a_envelope(body_type)

    new_config = agent.config.model_copy(deep=True)
    new_config.output_type = new_output_type

    # Create the new agent with the wrapped config, preserving MCP servers and tools
    return await agent._recreate_with_config_async(new_config)
