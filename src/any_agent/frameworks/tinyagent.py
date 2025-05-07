from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import litellm

from any_agent.config import AgentConfig, AgentFramework, TracingConfig
from any_agent.logging import logger
from any_agent.tracing.trace import AgentTrace

from .any_agent import AnyAgent

if TYPE_CHECKING:
    from collections.abc import Callable

    from litellm.types.utils import Message as LiteLLMMessage


DEFAULT_SYSTEM_PROMPT = """
You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved, or if you need more info from the user to solve the problem.

If you are not sure about anything pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.
""".strip()

# Max number of tool calling + chat completion steps in response to a single user query
DEFAULT_MAX_NUM_TURNS = 10


### Internal tools for tiny-agent ###"
def task_completion_tool() -> dict[str, Any]:
    """Tool to indicate task completion."""
    return {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Call this tool when the task given by the user is complete",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


class ToolExecutor:
    """Executor for tools that wraps tool functions to work with the MCP client."""

    def __init__(self, tool_function: Callable[..., Any]) -> None:
        """Initialize the tool executor.

        Args:
            tool_function: The tool function to execute

        """
        self.tool_function = tool_function

    async def call_tool(self, request: dict[str, Any]) -> dict[str, Any]:
        """Call the tool function.

        Args:
            request: The tool request with name and arguments

        Returns:
            Tool execution result

        """
        try:
            # Extract arguments
            arguments = request.get("arguments", {})

            # Call the tool function
            if asyncio.iscoroutinefunction(self.tool_function):
                result = await self.tool_function(**arguments)
            else:
                result = self.tool_function(**arguments)

            # Format the result
            if isinstance(result, str):
                return {"content": [{"text": result}]}
            if isinstance(result, dict):
                return {"content": [{"text": str(result)}]}
            return {"content": [{"text": str(result)}]}
        except Exception as e:
            return {"content": [{"text": f"Error executing tool: {e}"}]}


class TinyAgent(AnyAgent):
    """A lightweight agent implementation using litellm.

    Modeled after JS implementation https://huggingface.co/blog/tiny-agents.
    """

    def __init__(
        self,
        config: AgentConfig,
        managed_agents: list[AgentConfig] | None = None,
        tracing: TracingConfig | None = None,
    ) -> None:
        """Initialize the TinyAgent.

        Args:
            config: Agent configuration
            managed_agents: Optional list of managed agent configurations
            tracing: Optional tracing configuration

        """
        # we don't yet support multi-agent in tinyagent
        if managed_agents:
            msg = "Managed agents are not supported in TinyAgent."
            raise ValueError(msg)
        super().__init__(config, managed_agents=managed_agents, tracing=tracing)
        self.messages: list[dict[str, Any]] = []
        self.instructions = config.instructions or DEFAULT_SYSTEM_PROMPT
        self.api_key = config.api_key
        self.api_base = config.api_base
        self.model = config.model_id
        self.clients: dict[str, ToolExecutor] = {}
        self.available_tools: list[dict[str, Any]] = []
        self.exit_loop_tools = [task_completion_tool()]

    async def _load_agent(self) -> None:
        """Load the agent and its tools."""
        # Load tools
        logger.debug("Loading tools: %s", self.config.tools)
        wrapped_tools, mcp_servers = await self._load_tools(self.config.tools)
        self._mcp_servers = (
            mcp_servers  # Store servers so that they don't get garbage collected
        )
        logger.debug("Wrapped tools count: %s", len(wrapped_tools))

        for tool in wrapped_tools:
            tool_name = tool.__name__
            tool_desc = tool.__doc__ or f"Tool to {tool_name}"

            # check if the tool has __input__schema__ attribute which we set when wrapping MCP tools
            if not hasattr(tool, "__input_schema__"):
                # Generate one from the function signature
                import inspect

                sig = inspect.signature(tool)
                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    # Skip *args and **kwargs
                    if param.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue

                    # Add the parameter to properties
                    properties[param_name] = {
                        "type": "string",
                        "description": f"Parameter {param_name}",
                    }

                    # If parameter has no default, it's required
                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)

                input_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            else:
                # Use the provided schema
                input_schema = tool.__input_schema__

            # Add the tool to available tools
            self.available_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_desc,
                        "parameters": input_schema,
                    },
                }
            )

            # Register tool with the client
            self.clients[tool_name] = ToolExecutor(tool)
            logger.debug("Registered tool: %s", tool_name)

    async def run_async(self, prompt: str, **kwargs: Any) -> AgentTrace:
        """Run the agent asynchronously.

        Args:
            prompt: User input prompt
            **kwargs: Additional parameters

        Returns:
            The final agent response

        """
        self._setup_tracing()
        logger.debug("Running agent with prompt: %s...", prompt[:500])
        max_turns = kwargs.get("max_turns", DEFAULT_MAX_NUM_TURNS)
        self.messages = [
            {
                "role": "system",
                "content": self.instructions or DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        num_of_turns = 0
        next_turn_should_call_tools = True
        final_response = ""
        assistant_messages = []

        while True:
            try:
                logger.debug("Starting turn %s", num_of_turns + 1)

                last_response = await self._process_single_turn_with_tools(
                    {
                        "exit_loop_tools": self.exit_loop_tools,
                        "exit_if_first_chunk_no_tool": num_of_turns > 0
                        and next_turn_should_call_tools,
                    },
                )

                if last_response:
                    logger.debug(last_response)
                    logger.debug(
                        "Assistant response this turn: %s...",
                        last_response[:50],
                    )
                    assistant_messages.append(last_response)
                    final_response = last_response

            except Exception as err:
                logger.error("Error during turn %s: %s", num_of_turns + 1, err)
                if isinstance(err, Exception) and str(err) == "AbortError":
                    return AgentTrace(final_output=final_response or "Task aborted")
                raise

            num_of_turns += 1
            current_last = self.messages[-1]
            logger.debug("Current role: %s", current_last.get("role"))

            # After a turn, check if we have any content in the last assistant message
            if current_last.get("role") == "assistant" and current_last.get("content"):
                final_response = current_last.get("content", "No content found")
                logger.debug(
                    "Updated final response from assistant message: %s...",
                    final_response[:50],
                )

            # Check exit conditions
            if (
                current_last.get("role") == "tool"
                and current_last.get("name")
                and current_last.get("name")
                in [t["function"]["name"] for t in self.exit_loop_tools]
            ):
                logger.debug(
                    "Exiting because tool %s is an exit tool", current_last.get("name")
                )
                # If task is complete, return the last assistant message before this
                for msg in reversed(self.messages[:-1]):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        return AgentTrace(final_output=msg.get("content"))
                return AgentTrace(final_output=final_response or "Task completed")

            if current_last.get("role") != "tool" and num_of_turns > max_turns:
                logger.debug("Exiting because max turns (%s) reached", max_turns)
                return AgentTrace(
                    final_output=final_response or "Max turns reached",
                )

            if current_last.get("role") != "tool" and next_turn_should_call_tools:
                logger.debug("Exiting because no tools were called when expected")
                return AgentTrace(final_output=final_response or "No tools called")

            if current_last.get("role") == "tool":
                next_turn_should_call_tools = False
                logger.debug("Tool was called, next turn should not call tools")
            else:
                next_turn_should_call_tools = True
                logger.debug("No tool was called, next turn should call tools")

    async def _process_single_turn_with_tools(self, options: dict[str, Any]) -> str:
        """Process a single turn of conversation with potential tool calls.

        Args:
            options: Options including exit_loop_tools, exit_if_first_chunk_no_tool

        Returns:
            The response message or combined tool results

        """
        logger.debug("Start of single turn")

        tools = options["exit_loop_tools"] + self.available_tools

        # Create the completion parameters
        completion_params = {
            "model": self.model,
            "messages": self.messages,
            "tools": tools,
            "tool_choice": "auto",
        }

        # Add API key and base if provided
        if self.api_key:
            completion_params["api_key"] = self.api_key
        if self.api_base:
            completion_params["api_base"] = self.api_base
        response = await litellm.acompletion(**completion_params)
        message: LiteLLMMessage = response.choices[0].message
        logger.debug("Response message: %s", message.model_dump())
        self.messages.append(message.model_dump())

        # Process tool calls if any
        combined_results = []
        exit_tool_called = False

        if message.tool_calls:
            logger.debug(f"Processing {len(message.tool_calls)} tool calls")

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                logger.debug("Processing tool call for: %s", tool_name)
                tool_args = {}

                if tool_call.function.arguments:
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.debug("Tool arguments: %s", tool_args)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "",
                    "name": tool_name,
                }

                # Check if tool is in exit loop tools
                exit_tools = options["exit_loop_tools"]
                if exit_tools and tool_name in [
                    t["function"]["name"] for t in exit_tools
                ]:
                    logger.debug("Exit tool called: %s", tool_name)
                    exit_tool_called = True
                    self.messages.append(tool_message)
                    combined_results.append(str(tool_message["content"]))
                    continue

                # Check if the tool exists
                if tool_name not in self.clients:
                    logger.error("Tool %s not found in registered tools", tool_name)
                    tool_message["content"] = (
                        f"Error: No tool found with name: {tool_name}"
                    )
                else:
                    client = self.clients[tool_name]
                    try:
                        logger.debug("Calling tool: %s", tool_name)
                        result = await client.call_tool(
                            {"name": tool_name, "arguments": tool_args}
                        )

                        if (
                            isinstance(result, dict)
                            and "content" in result
                            and isinstance(result["content"], list)
                        ):
                            tool_message["content"] = result["content"][0]["text"]
                        else:
                            tool_message["content"] = str(result)

                        logger.debug(
                            "Tool result: %s...",
                            tool_message["content"][:50]
                            if tool_message["content"]
                            else "Empty",
                        )
                    except Exception as e:
                        logger.error("Error calling tool %s: %s", tool_name, e)
                        tool_message["content"] = f"Error calling tool {tool_name}: {e}"

                self.messages.append(tool_message)
                combined_results.append(str(tool_message["content"]))

            # If an exit tool was called, return early with the combined results
            if exit_tool_called:
                return "\n".join(combined_results)

            return "\n".join(combined_results)

        return str(message.content)

    @property
    def framework(self) -> AgentFramework:
        return AgentFramework.TINYAGENT
