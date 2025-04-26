"""MCP adapter for Tiny framework."""

from typing import Any, Optional

from any_agent.tools.mcp.mcp_server import MCPServerBase


class TinyMCPServer(MCPServerBase):
    """MCP adapter for Tiny framework."""

    mcp_available: bool = True
    libraries: str = "litellm"

    def _check_dependencies(self) -> None:
        """Check if the required libraries are installed."""
        try:
            import litellm  # noqa: F401
        except ImportError:
            self.mcp_available = False
            super()._check_dependencies()

    async def _setup_tools(self) -> None:
        """Set up the MCP tools for TinyAgent."""
        from any_agent.frameworks.tiny_agent import McpServerConnection

        # Create a connection to the MCP server
        if hasattr(self.mcp_tool, "command"):
            # This is MCPStdioParams
            server_params = {
                "command": self.mcp_tool.command,
                "args": self.mcp_tool.args,
                "tools": self.mcp_tool.tools,
                "timeout": self.mcp_tool.client_session_timeout_seconds,
            }
        else:
            # This is MCPSseParams
            server_params = {
                "url": self.mcp_tool.url,
                "headers": self.mcp_tool.headers,
                "tools": self.mcp_tool.tools,
                "timeout": self.mcp_tool.client_session_timeout_seconds,
            }

        # Create an MCP server connection
        server = McpServerConnection(server_params)
        await server.connect()
        
        # Get available tools
        tools_result = await server.list_tools()
        
        # Filter tools if specific tools were requested
        available_tools = self._filter_tools(tools_result.get("tools", []))
        
        # Create callable tool functions
        tool_list = []
        for tool_info in available_tools:
            tool_list.append(self._create_tool_from_info(tool_info, server))
            
        # Store tools as a list
        self.tools = tool_list
    
    def _create_tool_from_info(self, tool_info: dict, server: Any) -> callable:
        """Create a tool function from tool information."""
        tool_name = tool_info.get("name", "")
        tool_description = tool_info.get("description", "")
        
        async def tool_function(*args, **kwargs) -> Any:
            """Tool function that calls the MCP server."""
            # Combine args and kwargs
            combined_args = {}
            if args and len(args) > 0:
                combined_args = args[0]
            combined_args.update(kwargs)
            
            # Call the tool
            result = await server.call_tool({
                "name": tool_name,
                "arguments": combined_args
            })
            
            # Return the result
            if isinstance(result, dict) and "content" in result:
                if isinstance(result["content"], list) and len(result["content"]) > 0:
                    return result["content"][0].get("text", "")
                return str(result["content"])
            return str(result)
            
        # Set attributes for the tool function
        tool_function.__name__ = tool_name
        tool_function.__doc__ = tool_description
        
        return tool_function 