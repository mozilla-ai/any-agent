from typing import TYPE_CHECKING, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart
from a2a.utils import (
    new_agent_parts_message,
    new_task,
)
from pydantic import BaseModel

from any_agent.logging import logger
from any_agent.serving.envelope import A2AEnvelope
from any_agent.serving.task_manager import TaskManager

if TYPE_CHECKING:
    from any_agent import AnyAgent


class AnyAgentExecutor(AgentExecutor):  # type: ignore[misc]
    """AnyAgentExecutor Implementation with task management for multi-turn conversations."""

    def __init__(self, agent: "AnyAgent", task_manager: TaskManager):
        """Initialize the AnyAgentExecutor.

        Args:
            agent: The agent to execute
            task_manager: Task manager to use for task management

        """
        self.agent = agent
        self.task_manager = task_manager

    @override
    async def execute(  # type: ignore[misc]
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        # Extract or create task ID
        if not task:
            task = new_task(context.message)
            self.task_manager.add_task(task.id)
            await event_queue.enqueue_event(task)
        else:
            logger.debug("Task already exists: %s", task.model_dump_json(indent=2))

        formatted_query = self.task_manager.format_query_with_history(task.id, query)

        # This agent always produces Task objects.
        agent_trace = await self.agent.run_async(formatted_query)

        # Update task with new trace
        self.task_manager.update_task_trace(task.id, agent_trace)

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

        # Right now all task states will mark the state as final, because the server does not support streaming.
        # As we expand logic for streaming we may not need to always mark the state as final.
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
