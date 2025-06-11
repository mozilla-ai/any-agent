from typing import TYPE_CHECKING, Literal, Protocol, cast, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import (
    new_agent_parts_message,
    new_task,
)
from pydantic import BaseModel, ConfigDict

from any_agent.logging import logger

if TYPE_CHECKING:
    from any_agent import AnyAgent


class _DefaultBody(BaseModel):
    """Default payload when the user does not supply one."""

    result: str

    model_config = ConfigDict(extra="forbid")


# Accepted task_status values for output envelopes
SupportedTaskStatuses = Literal["complete", "input_required"]


# Protocol to help static typing recognize envelope attributes
class _OutputEnvelope(Protocol):
    task_status: SupportedTaskStatuses
    data: BaseModel


class AnyAgentExecutor(AgentExecutor):  # type: ignore[misc]
    """Test AgentProxy Implementation."""

    def __init__(self, agent: "AnyAgent"):
        """Initialize the AnyAgentExecutor."""
        self.agent = agent
        self._setup_output_type()

    def _setup_output_type(self) -> None:
        """Create a single envelope model."""
        body_type: type[BaseModel]

        if self.agent.config.output_type is None:
            body_type = _DefaultBody
        else:
            body_type = self.agent.config.output_type

        # Ensure the body model itself forbids extra keys so that its schema
        # carries `"additionalProperties": False`, a hard requirement for the
        # OpenAI LLM `response_format` parameter.
        if hasattr(body_type, "model_config"):
            body_type.model_config["extra"] = "forbid"
        else:
            body_type.model_config = ConfigDict(extra="forbid")

        # Build an envelope model dynamically so that its schema can still be
        # surfaced to downstream frameworks (they read config.output_type).
        # This makes the model give back the info we need in order to create the A2A status.
        class OutputContainer(BaseModel):
            """Output container for the agent."""

            task_status: SupportedTaskStatuses
            data: body_type  # type: ignore[valid-type]

            model_config = ConfigDict(extra="forbid")

        OutputContainer.__name__ = f"{body_type.__name__}Return"
        OutputContainer.__qualname__ = f"{body_type.__qualname__}Return"

        self.agent.config.output_type = OutputContainer

    @override
    async def execute(  # type: ignore[misc]
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        # This agent always produces Task objects.
        agent_trace = await self.agent.run_async(query)
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        else:
            logger.info("Task already exists: %s", task)
        updater = TaskUpdater(event_queue, task.id, task.contextId)

        # Validate & interpret the envelope produced by the agent
        final_output = agent_trace.final_output

        if not isinstance(final_output, BaseModel):  # pragma: no cover
            msg = f"Expected BaseModel, got {type(final_output)}"
            raise TypeError(msg)

        # Runtime attributes guaranteed by the dynamically created model.
        if not (hasattr(final_output, "task_status") and hasattr(final_output, "data")):
            msg = "Final output must have `task_status` and `data` attributes"
            raise AttributeError(msg)

        envelope = cast("_OutputEnvelope", final_output)
        task_status = envelope.task_status
        data_field = envelope.data

        # Convert payload to text we can stream to user
        if isinstance(data_field, _DefaultBody):
            result_text = data_field.result
        elif isinstance(data_field, BaseModel):
            result_text = data_field.model_dump_json()
        else:
            result_text = str(data_field)

        if task_status == "input_required":
            await updater.update_status(
                TaskState.input_required,
                new_agent_parts_message(
                    [Part(root=TextPart(text=result_text))],
                    task.contextId,
                    task.id,
                ),
                final=True,
            )
        else:
            # Same as calling updater.complete()
            await updater.update_status(
                TaskState.completed,
                message=new_agent_parts_message(
                    [Part(root=TextPart(text=result_text))],
                    task.contextId,
                    task.id,
                ),
                final=True,
            )

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[misc]
        msg = "cancel not supported"
        raise ValueError(msg)
