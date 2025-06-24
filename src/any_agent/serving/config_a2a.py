from collections.abc import Callable

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
        f"Current user message: {current_query}\n"
        "Please respond taking into account the conversation history above."
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
            task_timeout_minutes=15
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

    task_timeout_minutes: int = 10
    """Task timeout in minutes. Tasks will be cleaned up after this period of inactivity."""

    history_formatter: HistoryFormatter = default_history_formatter
    """Function to format conversation history and current query into a single prompt.
    Takes (messages, current_query) and returns formatted string."""

    task_cleanup_interval_minutes: int = 5
    """Interval in minutes between task cleanup runs."""
