# Agent Tools

`any-agent` provides 2 options to specify what `tools` are available to your agent: `Callables` and `MCP` ([Model Context Protocol](https://modelcontextprotocol.io/introduction)).

!!! tip

    Multi-agent can be implemented [using Agents-As-Tools](https://mozilla-ai.github.io/any-agent/agents/tools/#using-agents-as-tools).

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
    model_id="mistral:mistral-small-latest",
    tools=[search_web]
)
```

### Composio

We have a custom [Composio](https://github.com/ComposioHQ/composio) provider that allows to use
any [Composio tools](https://docs.composio.dev/toolkits/introduction) in `any-agent`:

!!! tip

    Check the different options for [Fetching and Filtering Composio Tools](https://docs.composio.dev/docs/fetching-tools).


```python
from any_agent import AgentConfig
from any_agent.tools.composio import CallableProvider
from composio import Composio

cpo = Composio(CallableProvider())

main_agent = AgentConfig(
    model_id="mistral:mistral-small-latest",
    tools=cpo.tools.get(
        user_id="daavoo",
        toolkits=["GITHUB", "HACKERNEWS"],
    )
)
```

### Using Agents-As-Tools

To directly use one agent as a tool for another agent, `any-agent` provides 3 different approaches:

The agent to be used as a tool can be defined as usual:

```py
from any_agent import AgentConfig, AnyAgent
from any_agent.tools import search_web

google_agent = await AnyAgent.create_async(
    "google",
    AgentConfig(
        name="google_expert",
        model_id="mistral:mistral-small-latest",
        instructions="Use the available tools to answer questions about the Google ADK",
        description="An agent that can answer questions about the Google Agents Development Kit (ADK).",
        tools=[search_web]
    )
)
```

You can then choose to wrap the agent using different approaches:

=== "Agent as Callable"

    ```py
    async def google_agent_as_tool(query: str) -> str:
        agent_trace = await google_agent.run_async(prompt=query)
        return str(agent_trace.final_output)

    google_agent_as_tool.__doc__ = google_agent.config.description
    ```

=== "Agent as MCP"

    ```py
    from any_agent.config import MCPSse
    from any_agent.serving import MCPServingConfig

    mcp_handle = await google_agent.serve_async(
        MCPServingConfig(port=5001, endpoint="/google-agent"))

    google_agent_as_tool = MCPSse(
        url="http://localhost:5001/google-agent/sse")
    ```

=== "Agent as A2A"

    ```py
    from any_agent.serving import A2AServingConfig
    from any_agent.tools import a2a_tool_async

    a2a_handle = await google_agent.serve_async(
        A2AServingConfig(port=5001, endpoint="/google-agent"))

    google_agent_as_tool = await a2a_tool_async(
        url="http://localhost:5001/google-agent")
    ```

Finally, regardless of the option chosen above, you can pass the agent as a tool to another agent:

```py
main_agent = await AnyAgent.create_async(
    "tinyagent",
    AgentConfig(
        name="main_agent",
        model_id="mistral-small-latest",
        instructions="Use the available tools to obtain additional information to answer the query.",
        tools=[google_agent_as_tool],
    )
)
```

## MCP

MCP can either be run locally ([MCPStdio][any_agent.config.MCPStdio]) or you can connect to an MCP that is running elsewhere (using either [MCPSse][any_agent.config.MCPSse] or [MCPStreamableHttp][any_agent.config.MCPStreamableHttp]).

!!! tip

    There are tools like [SuperGateway](https://github.com/supercorp-ai/supergateway) providing an easy way to turn a Stdio server into an SSE server.

!!! warning

    The SSE remote transport has been deprecated as of [MCP specification version 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http ). Please use the HTTP Stream Transport instead.

=== "MCP (Stdio)"

    See the [MCPStdio][any_agent.config.MCPStdio] API Reference.

    ```python
    from any_agent import AgentConfig
    from any_agent.config import MCPStdio

    main_agent = AgentConfig(
        model_id="mistral:mistral-small-latest",
        tools=[
            MCPStdio(
                command="docker",
                args=["run", "-i", "--rm", "mcp/fetch"],
                tools=["fetch"]
            ),
        ]
    )
    ```

=== "MCP (Streamable HTTP)"

    See the [MCPStreamableHttp][any_agent.config.MCPStreamableHttp] API Reference.

    ```python
    from any_agent import AgentConfig
    from any_agent.config import MCPStreamableHttp

    main_agent = AgentConfig(
        model_id="mistral:mistral-small-latest",
        tools=[
            MCPStreamableHttp(
                url="http://localhost:8000/mcp"
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
        model_id="mistral:mistral-small-latest",
        tools=[
            MCPSse(
                url="http://localhost:8000/sse"
            ),
        ]
    )
    ```

## Resource Management

When using tools where the python process running the agent is also responsible for managing the lifetime of the tool process (e.g., `MCPStdio`), the agent creates connections that should be properly cleaned up in order to avoid error messages like `RuntimeError: Attempted to exit cancel scope in a different task than it was entered in`.

### Using Context Manager (Recommended)

The recommended approach is to use the async context manager pattern, which automatically handles cleanup:

```python
import asyncio
from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStdio

async def main():
    time_tool = MCPStdio(
        command="uvx",
        args=["mcp-server-time"],
    )

    async with await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id="mistral:mistral-small-latest",
            tools=[time_tool],
        ),
    ) as agent:
        result = await agent.run_async("What time is it?")
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Manual Cleanup

If you can't use a context manager, you can manually clean up resources:

```python
import asyncio
from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStdio

async def main():
    time_tool = MCPStdio(
        command="uvx",
        args=["mcp-server-time"],
    )

    agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id="mistral:mistral-small-latest",
            tools=[time_tool],
        ),
    )

    try:
        result = await agent.run_async("What time is it?")
        print(result.final_output)
    finally:
        await agent.cleanup_async()

if __name__ == "__main__":
    asyncio.run(main())
```
