# adapted from https://github.com/google/a2a-python/blob/main/examples/helloworld/test_client.py

import re
from collections.abc import Callable, Coroutine
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from a2a.types import AgentCard

a2a_tool_available = False
with suppress(ImportError):
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

    a2a_tool_available = True


async def a2a_tool(
    url: str, toolname: str | None = None, http_kwargs: dict[str, Any] | None = None
) -> Callable[[str], Coroutine[Any, Any, str]]:
    """Perform a query using A2A to another agent.

    Args:
        url (str): The url in which the A2A agent is located.
        toolname (str): The name for the created tool. Defaults to `call_{agent name in card}`.
            Leading and trailing whitespace are removed. Whitespace in the middle is replaced by `_`.
        http_kwargs (dict): Additional kwargs to pass to the httpx client.

    Returns:
        An async `Callable` that takes a query and returns the agent response.

    """
    if not a2a_tool_available:
        msg = "You need to `pip install 'any-agent[a2a]'` to use this tool"
        raise ImportError(msg)

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
            response = await client.send_message(
                send_message_payload, http_kwargs=http_kwargs
            )
            result: str = response.model_dump_json()
            return result

    new_name = toolname or a2a_agent_card.name
    new_name = re.sub(r"\s+", "_", new_name.strip())
    _send_query.__name__ = f"call_{new_name}"
    _send_query.__doc__ = f"""{a2a_agent_card.description}
        Send a query to the agent named {a2a_agent_card.name}.

        Agent description: {a2a_agent_card.description}

        Args:
            query (str): The query to perform.

        Returns:
            The result from the A2A agent, encoded in json.
    """
    return _send_query
