# adapted from https://github.com/google/a2a-python/blob/main/examples/helloworld/test_client.py

import re
from collections.abc import Callable, Coroutine
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from any_agent.utils.asyncio_sync import run_async_in_sync

if TYPE_CHECKING:
    from a2a.types import AgentCard

a2a_tool_available = False
with suppress(ImportError):
    import httpx
    from a2a.client import A2ACardResolver, A2AClient
    from a2a.types import (
        JSONRPCErrorResponse,
        Message,
        MessageSendParams,
        Part,
        Role,
        SendMessageRequest,
        SendMessageSuccessResponse,
        Task,
        TextPart,
    )

    a2a_tool_available = True


async def a2a_tool_async(
    url: str, toolname: Optional[str] = None, http_kwargs: dict[str, Any] | None = None
) -> Callable[[str, Optional[str], Optional[str]], Coroutine[Any, Any, dict[str, Any]]]:
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

    if http_kwargs is None:
        http_kwargs = {}

    # Default timeout in httpx is 5 seconds. For an agent response, the default should be more lenient.
    if "timeout" not in http_kwargs:
        http_kwargs["timeout"] = 30.0

    async with httpx.AsyncClient(
        follow_redirects=True, **http_kwargs
    ) as resolver_client:
        a2a_agent_card: AgentCard = await (
            A2ACardResolver(httpx_client=resolver_client, base_url=url)
        ).get_agent_card()

    # NOTE: Use Optional[T] instead of T | None syntax throughout this module.
    # Google ADK's _parse_schema_from_parameter function has compatibility
    # with the traditional Optional[T] syntax for automatic function calling.
    # Using T | None syntax causes"Failed to parse the parameter ... for automatic function calling"
    async def _send_query(
        query: str, task_id: Optional[str] = None, context_id: Optional[str] = None
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(follow_redirects=True) as query_client:
            client = A2AClient(httpx_client=query_client, agent_card=a2a_agent_card)
            send_message_payload = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(
                    message=Message(
                        role=Role.user,
                        parts=[Part(root=TextPart(text=query))],
                        # the id is not currently tracked
                        messageId=str(uuid4().hex),
                        taskId=task_id,
                        contextId=context_id,
                    )
                ),
            )
            # TODO check how to capture exceptions and pass them on to the enclosing framework
            response = await client.send_message(
                send_message_payload, http_kwargs=http_kwargs
            )

            if not response.root:
                msg = (
                    "The A2A agent did not return a root. Are you using an A2A agent not managed by any-agent? "
                    "Please file an issue at https://github.com/mozilla-ai/any-agent/issues so we can help."
                )
                raise ValueError(msg)

            if isinstance(response.root, JSONRPCErrorResponse):
                response_dict = {
                    "error": response.root.error.message,
                    "code": response.root.error.code,
                    "data": response.root.error.data,
                }
            elif isinstance(response.root, SendMessageSuccessResponse):
                # Task
                if isinstance(response.root.result, Task):
                    task = response.root.result
                    response_dict = {
                        "timestamp": task.status.timestamp,
                        "status": task.status.state,
                    }
                    if task.status.message:
                        response_dict["task_id"] = task.status.message.taskId
                        response_dict["context_id"] = task.status.message.contextId
                        response_dict["message"] = {
                            " ".join(
                                [
                                    part.root.text
                                    for part in task.status.message.parts
                                    if isinstance(part.root, TextPart)
                                ]
                            )
                        }
                # Message
                else:
                    response_dict = {
                        "message": {
                            " ".join(
                                [
                                    part.root.text
                                    for part in response.root.result.parts
                                    if isinstance(part.root, TextPart)
                                ]
                            )
                        },
                        "task_id": response.root.result.taskId,
                    }
            else:
                msg = (
                    "The A2A agent did not return a error or a result. Are you using an A2A agent not managed by any-agent? "
                    "Please file an issue at https://github.com/mozilla-ai/any-agent/issues so we can help."
                )
                raise ValueError(msg)

            return response_dict

    new_name = toolname or a2a_agent_card.name
    new_name = re.sub(r"\s+", "_", new_name.strip())
    _send_query.__name__ = f"call_{new_name}"
    _send_query.__doc__ = f"""{a2a_agent_card.description}
        Send a query to the A2A hosted agent named {a2a_agent_card.name}.

        Agent description: {a2a_agent_card.description}

        Args:
            query (str): The query to send to the agent.
            task_id (str, optional): Task ID for continuing an incomplete task. Use the same
                task_id from a previous response with TaskState.input_required to resume the task. If you want to start a new task, you should not provide a task id.
            context_id (str, optional): Context ID for conversation continuity. Provides the
                agent with conversation history. Omit to start a fresh conversation. If you want to start a new conversation, you should not provide a context id.

        Returns:
            dict: Response from the A2A agent containing:
                - For successful responses: task_id, context_id, timestamp, status, and message
                - For errors: error message, code, and data

        Note:
            If TaskState is terminal (completed/failed), do not reuse the same task_id.
    """
    return _send_query


def a2a_tool(
    url: str, toolname: Optional[str] = None, http_kwargs: dict[str, Any] | None = None
) -> Callable[[str, Optional[str], Optional[str]], str]:
    """Perform a query using A2A to another agent (synchronous version).

    Args:
        url (str): The url in which the A2A agent is located.
        toolname (str): The name for the created tool. Defaults to `call_{agent name in card}`.
            Leading and trailing whitespace are removed. Whitespace in the middle is replaced by `_`.
        http_kwargs (dict): Additional kwargs to pass to the httpx client.

    Returns:
        A sync `Callable` that takes a query and returns the agent response.

    """
    if not a2a_tool_available:
        msg = "You need to `pip install 'any-agent[a2a]'` to use this tool"
        raise ImportError(msg)

    # Fetch the async tool upfront to get proper name and documentation (otherwise the tool doesn't have the right name and documentation)
    async_tool = run_async_in_sync(a2a_tool_async(url, toolname, http_kwargs))

    def sync_wrapper(
        query: str, task_id: Optional[str] = None, context_id: Optional[str] = None
    ) -> Any:
        """Execute the A2A tool query synchronously."""
        return run_async_in_sync(async_tool(query, task_id, context_id))

    # Copy essential metadata from the async tool
    sync_wrapper.__name__ = async_tool.__name__
    sync_wrapper.__doc__ = async_tool.__doc__

    return sync_wrapper
