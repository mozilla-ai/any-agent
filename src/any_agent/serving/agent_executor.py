from typing import TYPE_CHECKING, Protocol, cast, override

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


# Protocol to help static typing recognize envelope attributes
class _OutputEnvelope(Protocol):
    task_status: TaskState
    data: BaseModel


class AnyAgentExecutor(AgentExecutor):  # type: ignore[misc]
    """Test AgentProxy Implementation."""

    def __init__(self, agent: "AnyAgent"):
        """Initialize the AnyAgentExecutor."""
        self.agent = agent

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

        if not isinstance(final_output, BaseModel):
            msg = f"Expected BaseModel, got {type(final_output)}, {final_output}"
            raise TypeError(msg)

        # Runtime attributes guaranteed by the dynamically created model.
        if not (hasattr(final_output, "task_status") and hasattr(final_output, "data")):
            msg = "Final output must have `task_status` and `data` attributes"
            raise AttributeError(msg)

        envelope = cast("_OutputEnvelope", final_output)
        task_status = envelope.task_status
        data_field = envelope.data

        # Convert payload to text we can stream to user
        if isinstance(data_field, BaseModel):
            result_text = data_field.model_dump_json()
        else:
            result_text = str(data_field)

        # Right now all task states will mark the state as final. As we expand logic for multiturn tasks and streaming
        # we may not want to always mark the state as final.
        await updater.update_status(
            task_status,
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
