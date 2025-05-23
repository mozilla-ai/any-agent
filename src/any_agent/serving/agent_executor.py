from typing import TYPE_CHECKING, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

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
        agent_trace = await self.agent.run_async(query)
        assert agent_trace.final_output is not None
        event_queue.enqueue_event(new_agent_text_message(agent_trace.final_output))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # type: ignore[misc]
        msg = "cancel not supported"
        raise ValueError(msg)
