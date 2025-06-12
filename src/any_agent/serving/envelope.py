from typing import Generic, TypeVar

from a2a.types import TaskState
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound=BaseModel)


class A2AEnvelope(BaseModel, Generic[T]):
    """Base envelope class that all served agents must use for their output_type.

    This class enforces the A2A protocol requirements for agent outputs.
    Any agent that wants to be served must have an output_type that inherits from this class.

    Example:
        ```python
        class MyAgentOutput(BaseModel):
            result: str

        agent = AnyAgent.create(
            "openai",
            AgentConfig(
                model_id="gpt-4",
                output_type=A2AEnvelope[MyAgentOutput]
            )
        )
        ```

    """

    task_status: TaskState
    data: T

    model_config = ConfigDict(extra="forbid")


def validate_agent_envelope(output_type: type[BaseModel] | None) -> bool:
    """Validate that an agent's output_type is a valid A2A envelope.

    Args:
        output_type: The output_type from the agent's config

    Returns:
        bool: True if the output_type is a valid A2A envelope

    """
    if output_type is None:
        return False

    try:
        return issubclass(output_type, A2AEnvelope)
    except TypeError:
        return False
