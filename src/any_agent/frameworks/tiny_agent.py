from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union
import uuid

import litellm
from litellm.utils import StreamingChoices

from any_agent.config import AgentConfig, AgentFramework, Tool
from any_agent.logging import logger
from any_agent.frameworks.any_agent import AnyAgent

DEFAULT_SYSTEM_PROMPT = """
You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved, or if you need more info from the user to solve the problem.

If you are not sure about anything pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.
"""

# Max number of tool calling + chat completion steps in response to a single user query
MAX_NUM_TURNS = 10


def task_completion_tool() -> Dict[str, Any]:
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


def ask_question_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "ask_question",
            "description": "Ask a question to the user to get more info required to solve or clarify their problem.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


exit_loop_tools = [task_completion_tool(), ask_question_tool()]


class McpClient:
    """Model Context Protocol client for handling tool calls and model interactions."""

    def __init__(
        self, model: str, api_key: Optional[str] = None, api_base: Optional[str] = None
    ):
        """
        Initialize the MCP client

        Args:
            model: The model identifier
            api_key: API key for LLM service
            api_base: Base URL for API requests
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.clients = {}  # Map of tool name to tool client
        self.available_tools = []

    async def add_mcp_servers(self, servers: List[Dict[str, Any]]) -> None:
        """Add multiple MCP servers."""
        for server in servers:
            await self.add_mcp_server(server)

    async def add_mcp_server(self, server: Dict[str, Any]) -> None:
        """
        Add an MCP server with its tools

        Args:
            server: Server configuration parameters
        """
        # Create a client for the server
        client = McpServerConnection(server)
        await client.connect()

        # Get tools from server
        tools_result = await client.list_tools()
        logger.debug(
            f"Connected to server with tools: {[tool['name'] for tool in tools_result['tools']]}"
        )

        # Register tools with the client
        for tool in tools_result["tools"]:
            self.clients[tool["name"]] = client

        # Add tools to available tools list
        self.available_tools.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["inputSchema"],
                    },
                }
                for tool in tools_result["tools"]
            ]
        )

    async def process_single_turn_with_tools(
        self, messages: List[Dict[str, Any]], opts: Dict[str, Any] = None
    ) -> AsyncGenerator[Union[Dict[str, Any], Dict[str, str]], None]:
        """
        Process a single turn of conversation with potential tool calls
        
        Args:
            messages: List of conversation messages
            opts: Options including exit_loop_tools, exit_if_first_chunk_no_tool, and abort_signal
            
        Yields:
            Stream chunks or tool message results
        """
        if opts is None:
            opts = {}
            
        logger.debug("Start of single turn")
        
        # Log available tools
        tools_count = len(self.available_tools)
        logger.debug(f"Available tools count: {tools_count}")
        if tools_count > 0:
            tool_names = [t['function']['name'] for t in self.available_tools]
            logger.debug(f"Available tools: {tool_names}")
        
        # Configure tools based on options
        tools = (
            opts.get("exit_loop_tools", []) + self.available_tools
            if opts.get("exit_loop_tools")
            else self.available_tools
        )
        
        # Create the stream using litellm
        stream_params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        
        # Only add tools if we have some
        if tools:
            stream_params["tools"] = tools
            stream_params["tool_choice"] = "auto"
            logger.debug(f"Sending {len(tools)} tools to litellm")
        else:
            logger.debug("No tools available to send to model")
        
        # Add API key and base if provided
        if self.api_key:
            stream_params["api_key"] = self.api_key
        if self.api_base:
            stream_params["api_base"] = self.api_base
            
        # Create a new message to build up from stream chunks
        message = {
            "role": "assistant",
            "content": "",
        }
        
        final_tool_calls = {}
        num_of_chunks = 0
        
        try:
            logger.debug(f"Calling litellm.acompletion with model {self.model}")
            stream = await litellm.acompletion(**stream_params)
            
            async for chunk in stream:
                # Handle abort signal
                if opts.get("abort_signal"):
                    raise Exception("AbortError")
                    
                # Yield the chunk to the caller
                yield chunk
                
                num_of_chunks += 1
                
                # Process the delta from the chunk
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta if len(chunk.choices) > 0 else None
                
                if not delta:
                    continue
                    
                # Update message from delta
                if hasattr(delta, "role") and delta.role:
                    message["role"] = delta.role
                    
                if hasattr(delta, "content") and delta.content:
                    message["content"] += delta.content
                    
                # Process tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    logger.debug(f"Received tool calls in chunk {num_of_chunks}")
                    for tool_call in delta.tool_calls:
                        logger.debug(f"Tool call: {tool_call}")
                        # Initialize tool call if it's new
                        if tool_call.index not in final_tool_calls:
                            logger.debug(f"New tool call at index {tool_call.index}")
                            final_tool_calls[tool_call.index] = {
                                "id": tool_call.id if hasattr(tool_call, "id") else str(uuid.uuid4()),
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name if hasattr(tool_call.function, "name") else "",
                                    "arguments": ""
                                }
                            }
                            
                        # Add arguments if provided
                        if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                            final_tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
                
                # Check for early exit condition
                if (
                    opts.get("exit_if_first_chunk_no_tool")
                    and num_of_chunks <= 2
                    and len(final_tool_calls) == 0
                ):
                    return
            
            logger.debug(f"Received {num_of_chunks} chunks in total")
                    
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            if "AbortError" in str(e):
                raise Exception("AbortError")
            message["content"] = f"Error during model call: {str(e)}"
            final_tool_calls = {}
        
        # If we have content or tool calls, add the message to the conversation
        if message["content"] or final_tool_calls:
            if final_tool_calls:
                message["tool_calls"] = list(final_tool_calls.values())
                logger.debug(f"Added {len(final_tool_calls)} tool calls to message")
                
            messages.append(message)
            logger.debug(f"Added assistant message: {message['content'][:50]}...")
            
        # Log tool calls
        if final_tool_calls:
            logger.debug(f"Processing {len(final_tool_calls)} tool calls")
        
        # Process each tool call
        for tool_call in final_tool_calls.values():
            tool_name = tool_call["function"]["name"]
            logger.debug(f"Processing tool call for: {tool_name}")
            tool_args = {} 
            
            # Try to parse the arguments as JSON
            try:
                if tool_call["function"]["arguments"]:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    logger.debug(f"Tool arguments: {tool_args}")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse tool arguments for {tool_name}: {tool_call['function']['arguments']}")
            
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": "",
                "name": tool_name,
            }
            
            # Check if tool is in exit loop tools
            exit_tools = opts.get("exit_loop_tools", [])
            if exit_tools and tool_name in [t["function"]["name"] for t in exit_tools]:
                logger.debug(f"Exiting loop due to exit tool: {tool_name}")
                messages.append(tool_message)
                yield tool_message
                return
                
            # Check if the tool exists
            if tool_name not in self.clients:
                logger.error(f"Tool {tool_name} not found in registered tools")
                tool_message["content"] = f"Error: No tool found with name: {tool_name}"
                messages.append(tool_message)
                yield tool_message
                continue
                
            # Call the appropriate tool
            client = self.clients.get(tool_name)
            if client:
                try:
                    logger.debug(f"Calling tool: {tool_name}")
                    result = await client.call_tool(
                        {"name": tool_name, "arguments": tool_args}
                    )
                    logger.debug(f"Result type: {type(result)}")
                    
                    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
                        tool_message["content"] = result["content"][0]["text"]
                    else:
                        tool_message["content"] = str(result)
                    
                    logger.debug(f"Tool result: {tool_message['content'][:50]}...")
                except Exception as e:
                    logger.error(f"Error calling tool {tool_name}: {str(e)}")
                    tool_message["content"] = f"Error calling tool {tool_name}: {str(e)}"
            else:
                tool_message["content"] = (
                    f"Error: No session found for tool: {tool_name}"
                )
                
            messages.append(tool_message)
            yield tool_message

    async def cleanup(self) -> None:
        """Close all client connections."""
        clients = set(self.clients.values())
        for client in clients:
            await client.close()


class McpServerConnection:
    """Connection to an MCP server providing tools."""

    def __init__(self, server_params: Dict[str, Any]):
        """
        Initialize a connection to an MCP server

        Args:
            server_params: Server connection parameters
        """
        self.params = server_params
        self.tools = []

    async def connect(self) -> None:
        """Connect to the server."""
        # For a real implementation, this would connect to an actual MCP server
        # For simplicity, we'll set up some default tools
        self.tools = [
            {
                "name": "search",
                "description": "Search the web for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "read_file",
                "description": "Read a file from the filesystem",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        ]

    async def list_tools(self) -> Dict[str, Any]:
        """
        List tools available on the server

        Returns:
            Dictionary containing available tools
        """
        return {"tools": self.tools}

    async def call_tool(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the server

        Args:
            tool_request: Tool request with name and arguments

        Returns:
            Tool execution result
        """
        tool_name = tool_request.get("name", "")
        arguments = tool_request.get("arguments", {})
        
        try:
            if tool_name == "search":
                return await self._search(arguments.get("query", ""))
            elif tool_name == "read_file":
                return await self._read_file(arguments.get("path", ""))
            elif tool_name == "write_file":
                return await self._write_file(arguments.get("path", ""), arguments.get("content", ""))
            else:
                return {
                    "content": [
                        {"text": f"Unsupported tool: {tool_name}"}
                    ]
                }
        except Exception as e:
            return {
                "content": [
                    {"text": f"Error executing {tool_name}: {str(e)}"}
                ]
            }
    
    async def _search(self, query: str) -> Dict[str, Any]:
        """Simulate a web search"""
        return {
            "content": [
                {"text": f"Search results for '{query}':\n\n1. Result 1\n2. Result 2\n3. Result 3"}
            ]
        }
        
    async def _read_file(self, path: str) -> Dict[str, Any]:
        """Read a file from the filesystem"""
        try:
            with open(path, 'r') as f:
                content = f.read()
            return {
                "content": [
                    {"text": f"Content of {path}:\n\n{content}"}
                ]
            }
        except Exception as e:
            return {
                "content": [
                    {"text": f"Error reading file {path}: {str(e)}"}
                ]
            }
            
    async def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file"""
        try:
            with open(path, 'w') as f:
                f.write(content)
            return {
                "content": [
                    {"text": f"Successfully wrote {len(content)} characters to {path}"}
                ]
            }
        except Exception as e:
            return {
                "content": [
                    {"text": f"Error writing to file {path}: {str(e)}"}
                ]
            }

    async def close(self) -> None:
        """Close the connection to the server."""
        # Placeholder for connection cleanup
        pass


class ToolExecutor:
    """Executor for tools that wraps tool functions to work with the MCP client."""
    
    def __init__(self, tool_function: callable):
        """
        Initialize the tool executor
        
        Args:
            tool_function: The tool function to execute
        """
        self.tool_function = tool_function
        
    async def call_tool(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the tool function
        
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
                return {
                    "content": [
                        {"text": result}
                    ]
                }
            elif isinstance(result, dict):
                return {
                    "content": [
                        {"text": str(result)}
                    ]
                }
            else:
                return {
                    "content": [
                        {"text": str(result)}
                    ]
                }
        except Exception as e:
            return {
                "content": [
                    {"text": f"Error executing tool: {str(e)}"}
                ]
            }


class TinyAgent(AnyAgent):
    """A lightweight agent implementation using litellm."""

    def __init__(self, config: AgentConfig, managed_agents=None):
        """
        Initialize the TinyAgent

        Args:
            config: Agent configuration
            managed_agents: Optional list of managed agent configurations
        """
        super().__init__(config, managed_agents=managed_agents)
        self.messages = [
            {
                "role": "system",
                "content": config.instructions or DEFAULT_SYSTEM_PROMPT,
            }
        ]
        self.mcp_client = None

    async def load_agent(self) -> None:
        """Load the agent and its tools."""
        # Initialize the MCP client
        self.mcp_client = McpClient(
            model=self.config.model_id,
            api_key=self.config.api_key,
            api_base=self.config.api_base,
        )
        
        # Load tools
        logger.debug(f"Loading tools: {self.config.tools}")
        wrapped_tools, mcp_servers = await self._load_tools(self.config.tools)
        logger.debug(f"Wrapped tools count: {len(wrapped_tools)}")
        logger.debug(f"MCP servers count: {len(mcp_servers)}")
        
        # Register plain tools with the client
        for tool in wrapped_tools:
            try:
                tool_name = tool.__name__
                tool_desc = tool.__doc__ or f"Tool to {tool_name}"
                
                # Try to get input schema
                input_schema = getattr(tool, "input_schema", None)
                if not input_schema:
                    # Generate one from the function signature
                    import inspect
                    sig = inspect.signature(tool)
                    properties = {}
                    required = []
                    
                    for param_name, param in sig.parameters.items():
                        # Skip *args and **kwargs
                        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                            continue
                            
                        # Add the parameter to properties
                        properties[param_name] = {
                            "type": "string",
                            "description": f"Parameter {param_name}"
                        }
                        
                        # If parameter has no default, it's required
                        if param.default == inspect.Parameter.empty:
                            required.append(param_name)
                    
                    input_schema = {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                
                # Add the tool to available tools
                self.mcp_client.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_desc,
                        "parameters": input_schema
                    }
                })
                
                # Register tool with the client
                self.mcp_client.clients[tool_name] = ToolExecutor(tool)
                logger.debug(f"Registered tool: {tool_name}")
            except Exception as e:
                logger.error(f"Error registering tool: {str(e)}")
        
        # Register MCP servers with the client
        for server in mcp_servers:
            for tool in server.tools:
                try:
                    # Wrap the tool function to work with MCP client
                    tool_name = tool.__name__
                    tool_desc = tool.__doc__ or f"Tool to {tool_name}"
                    
                    # Add tool to the client's available tools
                    self.mcp_client.available_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_desc,
                            "parameters": getattr(tool, "input_schema", {
                                "type": "object",
                                "properties": {},
                                "required": []
                            })
                        }
                    })
                    
                    # Register the tool with the MCP client
                    self.mcp_client.clients[tool_name] = ToolExecutor(tool)
                    logger.debug(f"Registered MCP tool: {tool_name}")
                except Exception as e:
                    logger.error(f"Error registering MCP tool: {str(e)}")
                    
        # Log available tools
        logger.debug(f"Available tools: {[t['function']['name'] for t in self.mcp_client.available_tools]}")

    async def run_async(
        self, prompt: str, **kwargs
    ) -> Any:
        """
        Run the agent asynchronously

        Args:
            prompt: User input prompt
            **kwargs: Additional parameters like abort_signal

        Returns:
            The final agent response
        """
        logger.debug(f"Running agent with prompt: {prompt[:50]}...")
        
        self.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        num_of_turns = 0
        next_turn_should_call_tools = True
        final_response = ""
        assistant_messages = []

        while True:
            try:
                logger.debug(f"Starting turn {num_of_turns + 1}")
                chunks = []
                current_assistant_response = ""
                
                async for chunk in self.mcp_client.process_single_turn_with_tools(
                    self.messages,
                    {
                        "exit_loop_tools": exit_loop_tools,
                        "exit_if_first_chunk_no_tool": num_of_turns > 0
                        and next_turn_should_call_tools,
                        "abort_signal": kwargs.get("abort_signal"),
                    },
                ):
                    chunks.append(chunk)
                    # If this is a response chunk with content, add to current response
                    if (hasattr(chunk, "choices") and 
                        chunk.choices and 
                        hasattr(chunk.choices[0], "delta") and 
                        hasattr(chunk.choices[0].delta, "content") and
                        chunk.choices[0].delta.content):
                        current_assistant_response += chunk.choices[0].delta.content
                
                # If we got content from the assistant in this turn, add it to our collection
                if current_assistant_response:
                    logger.debug(f"Assistant response this turn: {current_assistant_response[:50]}...")
                    assistant_messages.append(current_assistant_response)
                    final_response = current_assistant_response
                
            except Exception as err:
                logger.error(f"Error during turn {num_of_turns + 1}: {str(err)}")
                if isinstance(err, Exception) and str(err) == "AbortError":
                    return final_response or "Task aborted"
                raise err

            num_of_turns += 1
            current_last = self.messages[-1]
            logger.debug(f"Current role: {current_last.get('role')}")
            
            # After a turn, check if we have any content in the last assistant message
            if current_last.get("role") == "assistant" and current_last.get("content"):
                final_response = current_last.get("content")
                logger.debug(f"Updated final response from assistant message: {final_response[:50]}...")

            # Check exit conditions
            if (
                current_last.get("role") == "tool"
                and current_last.get("name")
                and current_last.get("name")
                in [t["function"]["name"] for t in exit_loop_tools]
            ):
                logger.debug(f"Exiting because tool {current_last.get('name')} is an exit tool")
                # If task is complete, return the last assistant message before this
                for msg in reversed(self.messages[:-1]):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        return msg.get("content")
                return final_response or "Task completed"

            if current_last.get("role") != "tool" and num_of_turns > MAX_NUM_TURNS:
                logger.debug(f"Exiting because max turns ({MAX_NUM_TURNS}) reached")
                return final_response or f"Reached maximum number of turns ({MAX_NUM_TURNS})"

            if current_last.get("role") != "tool" and next_turn_should_call_tools:
                logger.debug("Exiting because no tools were called when expected")
                return final_response or "No tools called, stopping"

            if current_last.get("role") == "tool":
                next_turn_should_call_tools = False
                logger.debug("Tool was called, next turn should not call tools")
            else:
                next_turn_should_call_tools = True
                logger.debug("No tool was called, next turn should call tools")
                
        logger.debug("Exiting agent run loop")
        return final_response or "No response generated"

    @property
    def framework(self) -> AgentFramework:
        """The Agent Framework used."""
        return AgentFramework.TINY
