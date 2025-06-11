from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from any_agent.serving.agent_executor import _DefaultBody

if TYPE_CHECKING:
    from a2a.types import TaskState

    from any_agent import AnyAgent


def _is_a2a_envelope(typ: type[BaseModel] | None) -> bool:
    if typ is None:
        return False
    fields: Any = getattr(typ, "__fields__", None)

    # We only care about a mapping with the required keys.
    if not isinstance(fields, Mapping):
        return False

    return "task_status" in fields and "data" in fields


def _create_a2a_envelope(body_type: type[BaseModel]) -> type[BaseModel]:
    """Return a *new* Pydantic model that wraps *body_type* with TaskState + data."""
    # Ensure body forbids extra keys (OpenAI response_format requirement)
    if hasattr(body_type, "model_config"):
        body_type.model_config["extra"] = "forbid"
    else:
        body_type.model_config = ConfigDict(extra="forbid")

    class OutputContainer(BaseModel):
        task_status: TaskState
        data: body_type  # type: ignore[valid-type]

        model_config = ConfigDict(extra="forbid")

    OutputContainer.__name__ = f"{body_type.__name__}Return"
    OutputContainer.__qualname__ = f"{body_type.__qualname__}Return"
    return OutputContainer


def prepare_agent_for_a2a(agent: AnyAgent) -> AnyAgent:
    """Return an agent whose ``config.output_type`` is A2A-ready.

    If *agent* is already envelope-compatible we hand it back untouched.
    Otherwise we clone its config, wrap the output type, and spin up a
    *new* agent instance via `AnyAgent.create` so that framework-specific
    initialisation sees the correct schema right from the start.
    """
    if _is_a2a_envelope(agent.config.output_type):
        return agent

    body_type = agent.config.output_type or _DefaultBody
    new_output_type = _create_a2a_envelope(body_type)

    new_config = agent.config.model_copy(deep=True)
    new_config.output_type = new_output_type

    # Use the concrete agent class to recreate with the wrapped config
    return agent.__class__.create(agent.framework, new_config)


async def prepare_agent_for_a2a_async(agent: AnyAgent) -> AnyAgent:
    """Async counterpart of :pyfunc:`prepare_agent_for_a2a`."""
    if _is_a2a_envelope(agent.config.output_type):
        return agent

    body_type = agent.config.output_type or _DefaultBody
    new_output_type = _create_a2a_envelope(body_type)

    new_config = agent.config.model_copy(deep=True)
    new_config.output_type = new_output_type

    # Use the concrete agent class to recreate with the wrapped config
    return await agent.__class__.create_async(agent.framework, new_config)
