# Agent Tools

`any-agent` provides 2 options to specify what `tools` are available to your agent: `Callable`, or `MCP` ([Model Context Protocol](https://modelcontextprotocol.io/introduction)).

You can use any combination of options in the same agent.

Under the hood, `any-agent` takes care of wrapping the
tool so it becomes usable by the selected framework.

MCP can either be run locally (MCPStdio) or you can connect to an MCP that is running elsewhere (MCPSse).
See [SuperGateway](https://github.com/supercorp-ai/supergateway) for an easy way to turn a Stdio server into an SSE server.

=== "Callable"

    ```python
    from any_agent import AgentConfig
    from any_agent.tools import search_web

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        tools=[search_web]
    )
    ```

=== "MCP (Stdio)"

    ```python
    from any_agent import AgentConfig
    from any_agent.config import MCPStdioParams

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        tools=[
            MCPStdioParams(
                command="docker",
                args=["run", "-i", "--rm", "mcp/fetch"],
                tools=["fetch"]
            ),
        ]
    )
    ```

=== "MCP (SSE)"

    ```python
    from any_agent import AgentConfig
    from any_agent.config import MCPSseParams

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        tools=[
            MCPSseParams(
                url="http://localhost:8000/sse"
            ),
        ]
    )
    ```
