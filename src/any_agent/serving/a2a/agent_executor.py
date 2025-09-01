import builtins
import sys

PYTHONEGT312 = sys.version_info >= (3, 12)

from typing import TYPE_CHECKING, Any

if PYTHONEGT312:
    from typing import override
else:
    # Fix for Python 3.11
    # We will define a "noop" decorator that
    # returns the same function
    # Trying to modify decorators in place depending
    # on the python version is much more cumbersome
    from collections.abc import Callable
    from typing import Any, TypeVar

    # For any function that takes some params
    # and returns whatever (upper bound)....
    F = TypeVar("F", bound=Callable[..., Any])

    # ...we ensure that the decorator returns a function
    # with the same type constraints (basically,
    # because it's the same function), and that
    # the decorator doesn't require any extra info
    def override(func: F, /) -> F:  # noqa: D103
        return func


from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState, TextPart
from a2a.utils import (
    new_agent_parts_message,
    new_task,
)
from any_llm.utils.aio import run_async_in_sync
from pydantic import BaseModel

from any_agent import AgentRunError
from any_agent.callbacks.base import Callback
from any_agent.callbacks.context import Context
from any_agent.logging import logger
from any_agent.serving.a2a.context_manager import ContextManager
from any_agent.serving.a2a.envelope import A2AEnvelope

if TYPE_CHECKING:
    from any_agent import AnyAgent

INSIDE_NOTEBOOK = hasattr(builtins, "__IPYTHON__")


class _ToolUpdaterCallback(Callback):
    """Sends events to a taskupdater about tool execution."""

    def __init__(self, updater: TaskUpdater, context_id: str, task_id: str) -> None:
        self.context_id = context_id
        self.task_id = task_id
        self.updater = updater

    def before_tool_execution(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        payload = dict(getattr(context.current_span, "attributes", {}))
        run_async_in_sync(
            self.updater.update_status(
                TaskState.working,
                message=new_agent_parts_message(
                    [
                        Part(
                            root=DataPart(
                                data={
                                    "event_type": "tool_started",
                                    "payload": payload,
                                }
                            )
                        )
                    ],
                    self.context_id,
                    self.task_id,
                ),
                final=False,
            ),
            allow_running_loop=INSIDE_NOTEBOOK,
        )
        return context

    def after_tool_execution(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        """Will be called after any LLM Call is completed."""
        payload = dict(getattr(context.current_span, "attributes", {}))
        run_async_in_sync(
            self.updater.update_status(
                TaskState.working,
                message=new_agent_parts_message(
                    [
                        Part(
                            root=DataPart(
                                data={
                                    "event_type": "tool_finished",
                                    "payload": payload,
                                }
                            )
                        )
                    ],
                    self.context_id,
                    self.task_id,
                ),
                final=False,
            ),
            allow_running_loop=INSIDE_NOTEBOOK,
        )
        return context


class AnyAgentExecutor(AgentExecutor):
    """AnyAgentExecutor Implementation with task management for multi-turn conversations."""

    def __init__(
        self,
        agent: "AnyAgent",
        context_manager: ContextManager,
        stream_tool_usage: bool,
    ):
        """Initialize the AnyAgentExecutor.

        Args:
            agent: The agent to execute
            context_manager: context manager to use for context management
            stream_tool_usage: whether to stream tool execution results

        """
        self.agent = agent
        self.context_manager = context_manager
        self.stream_tool_usage = stream_tool_usage

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        # We will assume context.message will not be None
        context_id = context.message.context_id or ""  # type: ignore[union-attr]
        if not self.context_manager.get_context(context_id):
            self.context_manager.add_context(context_id)

        # Extract or create task ID
        if not task:
            if context.message is not None:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)
            else:
                msg = "Task does not exist but the message in context is None"
                logger.warning(msg)
                raise ValueError(msg)
        else:
            logger.info("Task already exists: %s", task.model_dump_json(indent=2))

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        tool_updater = _ToolUpdaterCallback(
            updater=updater, context_id=task.context_id, task_id=task.id
        )

        formatted_query = self.context_manager.format_query_with_history(
            context_id,
            query,
        )

        # Put a callback here to record tool calling and send updates
        if self.stream_tool_usage:
            self.agent.config.callbacks.append(tool_updater)

        # This agent always produces Task objects.
        try:
            agent_trace = await self.agent.run_async(formatted_query)
        except AgentRunError as e:
            logger.exception(f"Served request failed: {e!s}")
            agent_trace = e.trace

        # Remove the tool recording callback
        if self.stream_tool_usage:
            self.agent.config.callbacks.pop(-1)

        # Update task with new trace, passing the original query (not formatted)
        self.context_manager.update_context_trace(context_id, agent_trace, query)

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
                task.context_id,
                task.id,
            ),
            final=True,
        )

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = "cancel not supported"
        raise ValueError(msg)
