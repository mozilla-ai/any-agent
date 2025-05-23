# adapted from https://github.com/google/a2a-python/blob/main/examples/helloworld/test_client.py

from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TextPart,
)

if TYPE_CHECKING:
    from a2a.types import AgentCard


async def a2a_query(url: str) -> Callable[[str], Coroutine[Any, Any, str]]:
    """Perform a query using A2A to another agent.

    Args:
        url (str): The url in which the A2A agent is located.

    Returns:
        A Callable that takes a query and returns the agent response.

    """
    async with httpx.AsyncClient(follow_redirects=True) as resolver_client:
        a2a_agent_card: AgentCard = await (
            A2ACardResolver(httpx_client=resolver_client, base_url=url)
        ).get_agent_card()

    async def _send_query(query: str) -> str:
        async with httpx.AsyncClient(follow_redirects=True) as query_client:
            client = A2AClient(httpx_client=query_client, agent_card=a2a_agent_card)
            send_message_payload = SendMessageRequest(
                params=MessageSendParams(
                    message=Message(
                        role=Role.user,
                        parts=[Part(root=TextPart(text=query))],
                        # the id is not currently tracked
                        messageId=uuid4().hex,
                    )
                ),
                id=str(uuid4().hex),
            )
            # TODO check how to capture exceptions and pass them on to the enclosing framework
            response = await client.send_message(send_message_payload)
            result: str = response.model_dump_json()
            return result

    _send_query.__doc__ = f"""{a2a_agent_card.description}
        Send a query to the agent named {a2a_agent_card.name}.

        Agent description: {a2a_agent_card.description}

        Args:
            query (str): The query to perform.

        Returns:
            The result from the A2A agent, encoded in json.
    """
    return _send_query
