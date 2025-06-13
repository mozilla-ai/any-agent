from typing import TYPE_CHECKING, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    TextPart,
)
from a2a.utils import (
    new_agent_parts_message,
    new_task,
)
from pydantic import BaseModel

from any_agent.logging import logger
from any_agent.serving.envelope import A2AEnvelope

if TYPE_CHECKING:
    from any_agent import AnyAgent


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
        if not isinstance(final_output, A2AEnvelope):
            msg = "Final output must be an A2AEnvelope"
            raise TypeError(msg)

        task_status = final_output.task_status
        data_field = final_output.data

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
