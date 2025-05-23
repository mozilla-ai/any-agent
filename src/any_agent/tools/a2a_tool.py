# adapted from https://github.com/google/a2a-python/blob/main/examples/helloworld/test_client.py
import any_agent.serving

if any_agent.serving.a2a_serving_available:
    from common.client import A2ACardResolver, A2AClient
    from common.types import AgentCard

from collections.abc import Callable
from typing import Any
from uuid import uuid4


def a2a_query(url: str) -> Callable[[str], str]:
    """Perform a query using A2A to another agent.

    Args:
        query (str): The query to perform.

    Returns:
        The result from the A2A agent, encoded in json.

    """
    a2a_agent_card: AgentCard = (A2ACardResolver(base_url=url)).get_agent_card()
    print(str(a2a_agent_card))
    client = A2AClient(agent_card=a2a_agent_card)

    async def _send_query(query: str) -> str:
        send_message_payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": query}],
                # the id is not currently tracked
                # 'messageId': uuid4().hex,
            },
            "id": str(uuid4().hex),
        }
        response = await client.send_task(send_message_payload)
        return response.model_dump_json()

    _send_query.__doc__ = f"""{a2a_agent_card.description}
        Send a query to the agent named {a2a_agent_card.name}.

        Agent description: {a2a_agent_card.description}

        Args:
            query (str): The query to perform.

        Returns:
            The result from the A2A agent, encoded in json.
    """
    return _send_query
