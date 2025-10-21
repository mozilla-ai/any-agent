import asyncio
import time
from collections.abc import Sequence
from typing import Any

import httpx
import requests

from any_agent.config import AgentFramework
from any_agent.tracing.agent_trace import AgentSpan

DEFAULT_SMALL_MODEL_ID = "mistral:mistral-small-latest"

LLM_IMPORT_PATHS = {
    AgentFramework.GOOGLE: "any_agent.frameworks.google.acompletion",
    AgentFramework.LANGCHAIN: "any_agent.frameworks.langchain.acompletion",
    AgentFramework.TINYAGENT: "any_agent.frameworks.tinyagent.acompletion",
    AgentFramework.AGNO: "any_agent.frameworks.agno.acompletion",
    AgentFramework.OPENAI: "any_llm.AnyLLM.acompletion",
    AgentFramework.SMOLAGENTS: "any_llm.completion",
    AgentFramework.LLAMA_INDEX: "any_llm.AnyLLM.acompletion",
}


def get_default_agent_model_args(
    agent_framework: AgentFramework, model_id: str | None = None
) -> dict[str, Any]:
    """Get the default model arguments for an agent framework.

    Args:
        agent_framework (AgentFramework): The agent framework to get the default model arguments for.
        model_id (str, optional): The model ID to get specific model arguments for. Defaults to None.

    Returns:
        dict[str, Any]: The default model arguments for the agent framework.

    """
    model_args: dict[str, Any] = (
        {"parallel_tool_calls": False}
        if agent_framework not in [AgentFramework.AGNO, AgentFramework.LLAMA_INDEX]
        else {}
    )
    if agent_framework == AgentFramework.SMOLAGENTS:
        model_args["allow_running_loop"] = True

        if model_id == DEFAULT_SMALL_MODEL_ID:
            # For mistral-small-latest, the default tool call role conversions in smolagents do not work
            # See default here: https://github.com/huggingface/smolagents/blob/f76dee172666d7dad178aed06b257c629967733b/src/smolagents/models.py#L237
            from smolagents.models import MessageRole

            model_args["custom_role_conversions"] = {
                MessageRole.TOOL_CALL: MessageRole.USER,
                MessageRole.TOOL_RESPONSE: MessageRole.USER,
            }

    model_args["temperature"] = 0.0
    return model_args


DEFAULT_HTTP_KWARGS = {"timeout": 60.0}


def wait_for_server(
    server_url: str, max_attempts: int = 20, poll_interval: float = 0.5
) -> None:
    """Wait for a server to be ready.

    Args:
        server_url (str): The URL of the server to wait for.
        max_attempts (int): The maximum number of attempts to make.
        poll_interval (float): The interval between attempts.

    """
    attempts = 0
    while True:
        try:
            # Try to make a basic GET request to check if server is responding
            requests.get(server_url, timeout=1.0)
            return  # noqa: TRY300
        except (requests.RequestException, ConnectionError):
            # Server not ready yet, continue polling
            pass

        time.sleep(poll_interval)
        attempts += 1
        if attempts >= max_attempts:
            msg = f"Could not connect to {server_url}. Tried {max_attempts} times with {poll_interval} second interval."
            raise ConnectionError(msg)


async def wait_for_server_async(
    server_url: str, max_attempts: int = 20, poll_interval: float = 0.5
) -> None:
    """Wait for a server to be ready.

    Args:
        server_url (str): The URL of the server to wait for.
        max_attempts (int): The maximum number of attempts to make.
        poll_interval (float): The interval between attempts.

    """
    attempts = 0

    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Try to make a basic GET request to check if server is responding
                await client.get(server_url, timeout=1.0)
                return  # noqa: TRY300
            except (httpx.RequestError, httpx.TimeoutException):
                # Server not ready yet, continue polling
                pass

            await asyncio.sleep(poll_interval)
            attempts += 1
            if attempts >= max_attempts:
                msg = f"Could not connect to {server_url}. Tried {max_attempts} times with {poll_interval} second interval."
                raise ConnectionError(msg)


def group_spans(
    spans: Sequence[AgentSpan],
) -> tuple[Sequence[AgentSpan], Sequence[AgentSpan], Sequence[AgentSpan]]:
    """Group spans into agent invocations, llm calls and tool executions."""
    agent_invocations = []
    llm_calls = []
    tool_executions = []
    for span in spans:
        if span.is_agent_invocation():
            agent_invocations.append(span)
        elif span.is_llm_call():
            llm_calls.append(span)
        elif span.is_tool_execution():
            tool_executions.append(span)
        else:
            msg = f"Unexpected span: {span}"
            raise AssertionError(msg)
    return agent_invocations, llm_calls, tool_executions
