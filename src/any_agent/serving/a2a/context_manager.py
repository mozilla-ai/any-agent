from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from any_agent.logging import logger
from any_agent.tracing.agent_trace import AgentMessage, AgentTrace

if TYPE_CHECKING:
    from any_agent.serving import A2AServingConfig


class ContextData:
    """Data stored for each task."""

    def __init__(self, task_id: str):
        """Initialize task data.

        Args:
            task_id: Unique identifier for the task

        """
        self.task_id = task_id
        self.conversation_history: list[
            AgentMessage
        ] = []  # Store original user queries and responses as AgentMessage objects
        self.last_activity = datetime.now()
        self.created_at = datetime.now()

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()

    def is_expired(self, timeout_minutes: int) -> bool:
        """Check if the task has expired.

        Args:
            timeout_minutes: Timeout in minutes

        Returns:
            True if task is expired, False otherwise

        """
        expiration_time = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.now() > expiration_time


class ContextManager:
    """Manages agent conversation context for multi-turn interactions."""

    def __init__(self, config: "A2AServingConfig"):
        """Initialize the context manager.

        Args:
            config: Serving configuration containing context settings

        """
        self.config = config
        self._tasks: dict[str, ContextData] = {}
        self._last_cleanup: datetime | None = None

    def add_context(self, context_id: str) -> None:
        """Store a new context.

        This method will also trigger cleanup of expired context

        """
        self._cleanup_expired_contexts()

        self._tasks[context_id] = ContextData(context_id)
        logger.debug("Created new context: %s", context_id)

    def get_context(self, context_id: str) -> ContextData | None:
        """Get context data by ID.

        Args:
            context_id: context ID to retrieve

        Returns:
            ContextData if found and not expired, None otherwise

        """
        context = self._tasks.get(context_id)
        if not context:
            return None

        if context.is_expired(self.config.context_timeout_minutes):
            logger.debug("context %s expired, removing", context_id)
            self._tasks.pop(context_id, None)
            return None

        context.update_activity()
        return context

    def update_context_trace(
        self, context_id: str, agent_trace: AgentTrace, original_query: str
    ) -> None:
        """Update the agent trace for a context.

        Args:
            context_id: context ID to update
            agent_trace: New agent trace to merge/store
            original_query: The original user query (without history formatting)

        """
        context = self.get_context(context_id)
        if not context:
            logger.warning("Attempted to update non-existent context: %s", context_id)
            return

        messages = agent_trace.spans_to_messages()
        # Find the first user message and verify it contains the original query before updating
        first_user_index = None

        for i, message in enumerate(messages):
            if message.role == "user":
                first_user_index = i
                break

        if first_user_index is None:
            msg = "No user message found in trace."
            raise ValueError(msg)

        # Verify that the original query exists in the first user message to confirm it's the right one
        if original_query not in messages[first_user_index].content:
            msg = f"Original query '{original_query}' not found in first user message content."
            raise ValueError(msg)

        # Update the content of the first user message with the original query
        messages[first_user_index].content = original_query
        context.conversation_history.extend(messages)

        context.update_activity()

    def format_query_with_history(self, context_id: str, current_query: str) -> str:
        """Format a query with conversation history.

        Args:
            context_id: context ID to get history for
            current_query: Current user query

        Returns:
            Formatted query string with history context

        """
        context = self.get_context(context_id)
        if not context:
            return current_query

        # Use stored conversation history (already AgentMessage objects)
        history = context.conversation_history
        return self.config.history_formatter(history, current_query)

    def remove_context(self, context_id: str) -> None:
        """Remove a context.

        Args:
            context_id: context ID to remove

        """
        if context_id in self._tasks:
            del self._tasks[context_id]
            logger.info("Removed context: %s", context_id)

    def _cleanup_expired_contexts(self) -> None:
        """Clean up expired contexts."""
        expired_contexts = []

        for context_id, context in self._tasks.items():
            if context.is_expired(self.config.context_timeout_minutes):
                expired_contexts.append(context_id)

        for context_id in expired_contexts:
            self.remove_context(context_id)

        # Update last cleanup time
        self._last_cleanup = datetime.now()

        if expired_contexts:
            logger.info("Cleaned up %d expired contexts", len(expired_contexts))
        else:
            logger.debug("No expired contexts to clean up")
