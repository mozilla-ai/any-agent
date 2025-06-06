# Agent Tools

`any-agent` provides 2 options to specify what `tools` are available to your agent: `Callables` and `MCP` ([Model Context Protocol](https://modelcontextprotocol.io/introduction)). In order to support multi-agent systems, any agents served via A2A can also be integrated by wrapping the A2A connection in a callable function tool as described [below](#a2a-tools).

You can use any combination of options within the same agent.

## Callables

Any Python callable can be directly passed as tools.
You can define them in the same script, import it from an external package, etc.

Under the hood, `any-agent` takes care of wrapping the
tool so it becomes usable by the selected framework.

!!! tip

    Check all the [built-in callable tools](../api/tools.md) that any-agent provides.

```python
from any_agent import AgentConfig
from any_agent.tools import search_web

main_agent = AgentConfig(
    model_id="gpt-4o-mini",
    tools=[search_web]
)
```

## MCP
MCP can either be run locally ([MCPStdio][any_agent.config.MCPStdio]) or you can connect to an MCP that is running elsewhere ([MCPSse][any_agent.config.MCPSse]).

!!! tip

    There are tools like [SuperGateway](https://github.com/supercorp-ai/supergateway) providing an easy way to turn a Stdio server into an SSE server.

=== "MCP (Stdio)"

    See the [MCPStdio][any_agent.config.MCPStdio] API Reference.

    ```python
    from any_agent import AgentConfig
    from any_agent.config import MCPStdio

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        tools=[
            MCPStdio(
                command="docker",
                args=["run", "-i", "--rm", "mcp/fetch"],
                tools=["fetch"]
            ),
        ]
    )
    ```

=== "MCP (SSE)"

    See the [MCPSse][any_agent.config.MCPSse] API Reference.

    ```python
    from any_agent import AgentConfig
    from any_agent.config import MCPSse

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        tools=[
            MCPSse(
                url="http://localhost:8000/sse"
            ),
        ]
    )
    ```

## A2A tools

!!! tip

    More information about serving agents over the A2A protocol can be found [here](../serving.md)

`any-agent` provides a tool to wrap a connection to another another agent served over the A2A protocol, by invoking the `any_agent.tools.a2a_tool` or `any_agent.tools.a2a_tool_async` function, for example:

```python
import asyncio
from any_agent.tools import a2a_tool_async

async def main():
    some_agent_tool = await a2a_tool_async("http://example.net:10000/some_agent")

    agent_cfg = AgentConfig(
        instructions="Use the available tools to obtain additional information to answer the query.",
        description="A sample agent.",
        model_id="gpt-4o-mini",
        tools=[some_agent_tool],
    )
asyncio.run(main())
```

The tool description is derived from the agent card, which is retrieved when this function is invoked. View the docstring in [a2a_tool_async][any_agent.tools.a2a_tool_async] or [a2a_tool][any_agent.tools.a2a_tool] for a description of the arguments available.
