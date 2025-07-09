from collections.abc import Callable

from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.server.tasks.push_notification_config_store import (
    PushNotificationConfigStore,
)
from a2a.server.tasks.push_notification_sender import PushNotificationSender
from a2a.server.tasks.task_store import TaskStore
from a2a.types import AgentSkill
from pydantic import BaseModel, ConfigDict

from any_agent.tracing.agent_trace import AgentMessage

# Type alias for history formatting function
HistoryFormatter = Callable[[list[AgentMessage], str], str]


def default_history_formatter(messages: list[AgentMessage], current_query: str) -> str:
    """Format conversation history and current query into a single prompt.

    Args:
        messages: List of AgentMessage objects from spans_to_messages()
        current_query: The current user query

    Returns:
        Formatted prompt with conversation history

    """
    if not messages:
        return current_query

    # Convert previous conversation to readable format
    history_text = "\n".join(
        [
            f"{msg.role.capitalize()}: {msg.content}"
            for msg in messages
            if msg.role != "system"
        ]
    )

    return (
        f"Previous conversation:\n{history_text}\n"
        "Please respond to the current user message, taking into account the conversation history above."
        f"Current user message: {current_query}\n"
    )


class A2AServingConfig(BaseModel):
    """Configuration for serving an agent using the Agent2Agent Protocol (A2A).

    Example:
        config = A2AServingConfig(
            port=8080,
            endpoint="/my-agent",
            skills=[
                AgentSkill(
                    id="search",
                    name="web_search",
                    description="Search the web for information"
                )
            ],
            context_timeout_minutes=15
        )

    """

    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    """Will be passed as argument to `uvicorn.run`."""

    port: int = 5000
    """Will be passed as argument to `uvicorn.run`."""

    endpoint: str = "/"
    """Will be pass as argument to `Starlette().add_route`"""

    log_level: str = "warning"
    """Will be passed as argument to the `uvicorn` server."""

    skills: list[AgentSkill] | None = None
    """List of skills to be used by the agent.

    If not provided, the skills will be inferred from the tools.
    """

    version: str = "0.1.0"

    context_timeout_minutes: int = 10
    """Context timeout in minutes. Contexts will be cleaned up after this period of inactivity."""

    history_formatter: HistoryFormatter = default_history_formatter
    """Function to format conversation history and current query into a single prompt.
    Takes (messages, current_query) and returns formatted string."""

    task_cleanup_interval_minutes: int = 5
    """Interval in minutes between task cleanup runs."""

    push_notifier_store_type: type[PushNotificationConfigStore] = (
        InMemoryPushNotificationConfigStore
    )

    """Push notifier config store to be used by the agent.

    If not provided, a default in-memory push notifier config store will be used.
    """

    push_notifier_sender_type: type[PushNotificationSender] = BasePushNotificationSender
    """Push notifier sender to be used by the agent.

    If not provided, a default async httpx-based push notifier sender will be used.
    """

    task_store_type: type[TaskStore] = InMemoryTaskStore
    """Task store to be used by the agent.

    If not provided, a default in-memory task store will be used.
    """
