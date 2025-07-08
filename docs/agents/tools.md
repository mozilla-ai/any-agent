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
    model_id="mistral/mistral-small-latest",
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
        model_id="mistral/mistral-small-latest",
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
        model_id="mistral/mistral-small-latest",
        tools=[
            MCPSse(
                url="http://localhost:8000/sse"
            ),
        ]
    )
    ```

## Using Agents-As-Tools

To directly use one agent as a tool for another agent, `any-agent` provides 3 different approaches:

The agent to be used as a tool can be defined as usual:

```py
from any_agent import AgentConfig, AnyAgent
from any_agent.tools import search_web

google_agent = await AnyAgent.create_async(
    "google",
    AgentConfig(
        name="google_expert",
        model_id="mistral/mistral-small-latest",
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
