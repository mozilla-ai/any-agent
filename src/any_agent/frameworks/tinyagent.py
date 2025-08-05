from __future__ import annotations

import asyncio
import inspect
import json
import os
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import litellm
from litellm.utils import supports_response_schema
from mcp.types import CallToolResult, TextContent

from any_agent.config import AgentConfig, AgentFramework

from .any_agent import AnyAgent

if TYPE_CHECKING:
    from collections.abc import Callable

    from litellm.types.utils import Message as LiteLLMMessage
    from litellm.types.utils import ModelResponse
    from pydantic import BaseModel


DEFAULT_SYSTEM_PROMPT = """
You are an agent that uses tools (whenever they are available) to answer the user's query.

You will keep calling tools until you find a final answer.
Once you have a final answer, you MUST call the `final_answer` tool.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls.
""".strip()


DEFAULT_MAX_NUM_TURNS = 10


class ToolExecutor:
    """Executor for tools that wraps tool functions to work with the MCP client."""

    def __init__(self, tool_function: Callable[..., Any]) -> None:
        """Initialize the tool executor.

        Args:
            tool_function: The tool function to execute

        """
        self.tool_function = tool_function

    async def call_tool(self, request: dict[str, Any]) -> str:
        """Call the tool function.

        Args:
            request: The tool request with name and arguments

        Returns:
            Tool execution result

        """
        try:
            arguments = request.get("arguments", {})

            if hasattr(self.tool_function, "__annotations__"):
                func_args = self.tool_function.__annotations__
                for arg_name, arg_type in func_args.items():
                    if arg_name in arguments:
                        with suppress(Exception):
                            arguments[arg_name] = arg_type(arguments[arg_name])

            if asyncio.iscoroutinefunction(self.tool_function):
                result = await self.tool_function(**arguments)
            else:
                result = self.tool_function(**arguments)

            if (
                isinstance(result, CallToolResult)
                and result.content
                and isinstance(result.content[0], TextContent)
            ):
                result = result.content[0].text
            return str(result)

        except Exception as e:
            return f"Error executing tool: {e}"


def final_answer(answer: str) -> str:
    """Return the final answer to the user."""
    return answer


class TinyAgent(AnyAgent):
    """A lightweight agent implementation using litellm.

    Modeled after JS implementation https://huggingface.co/blog/tiny-agents.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the TinyAgent.

        Args:
            config: Agent configuration
            tracing: Optional tracing configuration

        """
        super().__init__(config)
        self.clients: dict[str, ToolExecutor] = {}

        self.completion_params = {
            "model": self.config.model_id,
            "tools": [],
            "tool_choice": "required",
            **(self.config.model_args or {}),
        }

        if self.completion_params["tool_choice"] == "required":
            self.config.tools.append(final_answer)

        if self.config.api_key:
            self.completion_params["api_key"] = self.config.api_key
        if self.config.api_base:
            self.completion_params["api_base"] = self.config.api_base

        # Initialize providers client if gateway provider is set
        self.use_any_llm = os.getenv("USE_ANY_LLM")

    async def _load_agent(self) -> None:
        """Load the agent and its tools."""
        wrapped_tools, mcp_servers = await self._load_tools(self.config.tools)
        self._mcp_servers = (
            mcp_servers  # Store servers so that they don't get garbage collected
        )

        self._tools = wrapped_tools

        for tool in wrapped_tools:
            tool_name = tool.__name__
            tool_desc = tool.__doc__ or f"Tool to {tool_name}"

            if not hasattr(tool, "__input_schema__"):
                sig = inspect.signature(tool)
                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    if param.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue

                    properties[param_name] = {
                        "type": "string",
                        "description": f"Parameter {param_name}",
                    }

                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)

                input_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            else:
                input_schema = tool.__input_schema__

            self.completion_params["tools"].append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_desc,
                        "parameters": input_schema,
                    },
                }
            )

            self.clients[tool_name] = ToolExecutor(tool)

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        messages = [
            {
                "role": "system",
                "content": self.config.instructions or DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        num_of_turns = 0
        max_turns = kwargs.get("max_turns", DEFAULT_MAX_NUM_TURNS)
        completion_params = self.completion_params.copy()

        while num_of_turns < max_turns:
            completion_params["messages"] = messages

            response = await self.call_model(**completion_params)

            message: LiteLLMMessage = response.choices[0].message  # type: ignore[union-attr]

            messages.append(message.model_dump())

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "",
                        "name": tool_name,
                    }

                    if tool_name not in self.clients:
                        tool_message["content"] = (
                            f"Error calling tool: No tool found with name: {tool_name}"
                        )
                        continue

                    tool_args = {}
                    if tool_call.function.arguments:
                        tool_args = json.loads(tool_call.function.arguments)

                    client = self.clients[tool_name]
                    result = await client.call_tool(
                        {"name": tool_name, "arguments": tool_args}
                    )
                    tool_message["content"] = result
                    messages.append(tool_message)  # type: ignore[arg-type]

                    if tool_name == "final_answer":
                        if self.config.output_type:
                            return await self._return_output_type(
                                str(result), completion_params
                            )
                        return str(result)

            elif message.role == "assistant" and message.content:
                if self.config.output_type:
                    return await self._return_output_type(
                        str(message.content), completion_params
                    )
                return str(message.content)

            num_of_turns += 1

        return "Max turns reached"

    async def _return_output_type(
        self, output: str, completion_params: dict[str, Any]
    ) -> str | BaseModel:
        if not self.config.output_type:
            return output
        completion_params["messages"] = [
            {
                "role": "system",
                "content": "You are an expert that can convert raw text into structured JSON.",
            },
            {
                "role": "user",
                "content": f"Please conform this output:\n{output}\nTo match the following schema:\n{self.config.output_type.model_json_schema()}.",
            },
        ]

        if self.use_any_llm or supports_response_schema(model=self.config.model_id):
            completion_params["response_format"] = self.config.output_type
        if "tools" in completion_params:
            completion_params.pop("tools")
            completion_params.pop("tool_choice", None)
            completion_params.pop("parallel_tool_calls", None)
        response = await self.call_model(**completion_params)
        if self.use_any_llm:
            return self.config.output_type.model_validate_json(
                response.choices[0].message.content  # type: ignore[arg-type, union-attr]
            )
        return self.config.output_type.model_validate_json(
            response.choices[0].message["content"]  # type: ignore[union-attr]
        )

    async def call_model(self, **completion_params: dict[str, Any]) -> ModelResponse:
        if self.use_any_llm:
            from any_llm import completion

            return completion(**completion_params)  # type: ignore[return-value, arg-type]
        # otherwise use litellm
        return await litellm.acompletion(**completion_params)  # type: ignore[no-any-return]

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        """Update the output type of the agent in-place.

        Args:
            output_type: The new output type to use, or None to remove output type constraint

        """
        self.config.output_type = output_type

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.TINYAGENT
