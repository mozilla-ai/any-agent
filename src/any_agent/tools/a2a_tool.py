# adapted from https://github.com/google/a2a-python/blob/main/examples/helloworld/test_client.py
import httpx
from typing import Any, Callable
from common.client import A2AClient
from uuid import uuid4

def a2a_query(url: str, description: str) -> Callable[[str], str]:
    """Perform a query using A2A to another agent.

    Args:
        query (str): The query to perform.

    Returns:
        The result from the A2A agent, encoded in json.

    """
    client = A2AClient(url=url)
    async def _send_query(query: str) -> str:
        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'text': query}
                ],
                # the id is not currently tracked
                # 'messageId': uuid4().hex,
            },
            'id': str(uuid4().hex),
        }
        response = await client.send_task(send_message_payload)
        return response.model_dump_json()

    _send_query.__doc__ = f"""{description}

        Args:
            query (str): The query to perform.

        Returns:
            The result from the A2A agent, encoded in json.
    """
    return _send_query
