from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from any_agent.logging import logger
from any_agent.tracing.agent_trace import AgentMessage, AgentTrace

if TYPE_CHECKING:
    from any_agent.serving import A2AServingConfig


class TaskData:
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


class TaskManager:
    """Manages agent conversation tasks for multi-turn interactions."""

    def __init__(self, config: "A2AServingConfig"):
        """Initialize the task manager.

        Args:
            config: Serving configuration containing task settings

        """
        self.config = config
        self._tasks: dict[str, TaskData] = {}
        self._last_cleanup: datetime | None = None

    def add_task(self, task_id: str) -> None:
        """Store a new task.

        This method will also trigger cleanup of expired tasks if it hasn't
        run recently (based on task_cleanup_interval_minutes).

        """
        self._cleanup_expired_tasks()

        self._tasks[task_id] = TaskData(task_id)
        logger.debug("Created new task: %s", task_id)

    def _get_task(self, task_id: str) -> TaskData | None:
        """Get task data by ID.

        Args:
            task_id: Task ID to retrieve

        Returns:
            TaskData if found and not expired, None otherwise

        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        if task.is_expired(self.config.task_timeout_minutes):
            logger.debug("Task %s expired, removing", task_id)
            self._tasks.pop(task_id, None)
            return None

        task.update_activity()
        return task

    def update_task_trace(
        self, task_id: str, agent_trace: AgentTrace, original_query: str
    ) -> None:
        """Update the agent trace for a task.

        Args:
            task_id: Task ID to update
            agent_trace: New agent trace to merge/store
            original_query: The original user query (without history formatting)

        """
        task = self._get_task(task_id)
        if not task:
            logger.warning("Attempted to update non-existent task: %s", task_id)
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
        task.conversation_history.extend(messages)

        task.update_activity()

    def format_query_with_history(self, task_id: str, current_query: str) -> str:
        """Format a query with conversation history.

        Args:
            task_id: Task ID to get history for
            current_query: Current user query

        Returns:
            Formatted query string with history context

        """
        task = self._get_task(task_id)
        if not task:
            return current_query

        # Use stored conversation history (already AgentMessage objects)
        history = task.conversation_history
        return self.config.history_formatter(history, current_query)

    def remove_task(self, task_id: str) -> None:
        """Remove a task.

        Args:
            task_id: Task ID to remove

        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            logger.info("Removed task: %s", task_id)

    def _cleanup_expired_tasks(self) -> None:
        """Clean up expired tasks."""
        expired_tasks = []

        for task_id, task in self._tasks.items():
            if task.is_expired(self.config.task_timeout_minutes):
                expired_tasks.append(task_id)

        for task_id in expired_tasks:
            self.remove_task(task_id)

        # Update last cleanup time
        self._last_cleanup = datetime.now()

        if expired_tasks:
            logger.info("Cleaned up %d expired tasks", len(expired_tasks))
        else:
            logger.debug("No expired tasks to clean up")
