from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from a2a.types import TaskState  # noqa: TC002
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from any_agent import AnyAgent


# Define a TypeVar for the body type
BodyType = TypeVar("BodyType", bound=BaseModel)


class A2AEnvelope(BaseModel, Generic[BodyType]):
    """A2A envelope that wraps response data with task status."""

    task_status: Literal[
        TaskState.input_required, TaskState.completed, TaskState.failed
    ]
    """Restricted to the states that are leveraged by our implementation of the A2A protocol.
    When we support streaming, the rest of the states can be added and supported."""

    data: BodyType

    model_config = ConfigDict(extra="forbid")


def _is_a2a_envelope(typ: type[BaseModel] | None) -> bool:
    """Check if a type is an A2A envelope."""
    if typ is None:
        return False
    fields: Any = getattr(typ, "model_fields", None)

    # We only care about a mapping with the required keys.
    if not isinstance(fields, Mapping):
        return False

    return "task_status" in fields and "data" in fields


def validate_a2a_output_type(agent: AnyAgent) -> None:
    """Validate that the agent's output_type is properly set up for A2A serving.

    Args:
        agent: The agent to validate.

    Raises:
        ValueError: If the output_type is not properly configured for A2A serving.

    """
    if not _is_a2a_envelope(agent.config.output_type):
        msg = "Agent output_type must inherit from A2AEnvelope for A2A serving. "
        msg += f"Current output_type: {agent.config.output_type}. "
        msg += (
            "Please set up your output_type to inherit from A2AEnvelope[YourDataType]."
        )
        raise ValueError(msg)
