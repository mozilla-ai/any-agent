from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from any_llm import completion
from any_llm.provider import ProviderFactory, ProviderName
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from any_agent.config import AgentConfig, AgentFramework
from any_agent.logging import logger

from .any_agent import AnyAgent

if TYPE_CHECKING:
    from collections.abc import Callable

    from any_llm.types.completion import ChatCompletion
    from pydantic import BaseModel


DEFAULT_SYSTEM_PROMPT = """
You are an agent that uses tools (whenever they are available) to answer the user's query.

You will keep calling tools until you find a final answer.
Once you have a final answer, you MUST call the `final_answer` tool.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls.
""".strip()

FINAL_ANSWER_TOOL = "provide_final_answer"


def _parse_output_type(
    output_type: type[BaseModel], answer: str | Any
) -> BaseModel | str:
    try:
        if isinstance(answer, str):
            return output_type.model_validate_json(answer)
        return output_type.model_validate(answer)
    except ValidationError:
        return str(answer)


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
            **(self.config.model_args or {}),
        }

        provider_name, _ = ProviderFactory.split_model_provider(self.config.model_id)
        self.uses_openai = provider_name == ProviderName.OPENAI

        user_tool_choice = self.completion_params.get("tool_choice")
        if user_tool_choice and user_tool_choice != "required" and not self.uses_openai:
            logger.warning(
                "TinyAgent framework only works with tool_choice=required."
                f"Provided option will be ignored: {user_tool_choice}"
            )

        if self.config.api_key:
            self.completion_params["api_key"] = self.config.api_key
        if self.config.api_base:
            self.completion_params["api_base"] = self.config.api_base

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
                    if param.default == inspect.Parameter.empty or self.uses_openai:
                        required.append(param_name)

                input_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            else:
                input_schema = tool.__input_schema__

            function_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": input_schema,
                },
            }
            if self.uses_openai:
                function_def["function"]["parameters"]["additionalProperties"] = False  # type: ignore[index]
                function_def["function"]["strict"] = True  # type: ignore[index]

            self.completion_params["tools"].append(function_def)

            self.clients[tool_name] = ToolExecutor(tool)

    def add_final_answer_tool(self) -> None:
        if self.config.output_type:
            description = (
                "Provide the final answer to the user. Must follow the schema:\n"
                f"{self.config.output_type.model_json_schema()}"
            )
        else:
            description = "Provide the final answer to the user."
        self.completion_params["tools"].append(
            {
                "type": "function",
                "function": {
                    "name": FINAL_ANSWER_TOOL,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": description}
                        },
                        "required": ["answer"],
                    },
                },
            }
        )

    async def _run_async(self, prompt: str, **kwargs: Any) -> str | BaseModel:
        if self.uses_openai:
            self.completion_params["tool_choice"] = "auto"
            if self.config.output_type is not None:
                self.completion_params["response_format"] = self.config.output_type
        else:
            self.completion_params["tool_choice"] = "required"
            self.add_final_answer_tool()

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

        if kwargs.pop("max_turns", None):
            logger.warning(
                "`max_turns` is deprecated and has no effect. See https://mozilla-ai.github.io/any-agent/agents/callbacks/#example-limit-the-number-of-steps"
            )
        completion_params = self.completion_params.copy()

        while True:
            completion_params["messages"] = messages

            response: ChatCompletion = await self.call_model(**completion_params)

            message = response.choices[0].message

            messages.append(message.model_dump())

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    f = tool_call.function  # type: ignore[union-attr]
                    tool_name = f.name

                    if tool_name == FINAL_ANSWER_TOOL:
                        if isinstance(f.arguments, str):
                            answer = json.loads(f.arguments)
                        else:
                            answer = f.arguments
                        if self.config.output_type:
                            return _parse_output_type(
                                self.config.output_type, answer.get("answer")
                            )
                        return str(answer.get("answer"))

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
                    if f.arguments:
                        tool_args = json.loads(f.arguments)

                    client = self.clients[tool_name]
                    result = await client.call_tool(
                        {"name": tool_name, "arguments": tool_args}
                    )
                    tool_message["content"] = result
                    messages.append(tool_message)

            elif message.role == "assistant" and message.content:
                if self.config.output_type:
                    return _parse_output_type(self.config.output_type, message.content)
                return str(message.content)

    async def call_model(self, **completion_params: dict[str, Any]) -> ChatCompletion:
        return completion(**completion_params)  # type: ignore[return-value, arg-type]

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
