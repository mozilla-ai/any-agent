# Serving

`any-agent` provides a simple way of serving agents from any of the supported frameworks using different protocols:


- [Agent2Agent Protocol (A2A)](https://google.github.io/A2A/), via the [A2A Python SDK](https://github.com/google-a2a/a2a-python). In order to this protocol, you must install the 'a2a' extra: `pip install 'any-agent[a2a]'`.

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/specification/2025-03-26), via the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).

- **MCP to A2A Bridge**: Expose MCP servers as A2A-compatible services, enabling protocol interoperability. Requires both extras: `pip install 'any-agent[mcp,a2a]'`.

## Configuring and Serving Agents

You can configure and serve an agent using the [`A2AServingConfig`][any_agent.serving.A2AServingConfig] or [`MCPServingConfig`][any_agent.serving.MCPServingConfig] and the [`AnyAgent.serve_async`][any_agent.AnyAgent.serve_async] method.

For illustrative purposes, we are going to define 2 separate scripts, each defining an agent to answer questions about a specific agent framework (either Google ADK or OpenAI Agents SDK):

!!! note

    We are using here the `google` and `openai` frameworks for each corresponding "expert" but we could actually use
    any of the supported frameworks, as the actual behavior is defined by the `instructions` and `description`.

=== "Google Expert"

    ```python
    # google_expert.py
    import argparse
    import asyncio

    from any_agent import AgentConfig, AnyAgent
    from any_agent.tools import search_web


    async def serve_agent(protocol):
        agent = await AnyAgent.create_async(
            "google",
            AgentConfig(
                name="google_expert",
                model_id="mistral/mistral-small-latest",
                instructions="You can answer questions about the Google Agents Development Kit (ADK) but nothing else",
                description="An agent that can answer questions specifically and only about the Google Agents Development Kit (ADK).",
                tools=[search_web]
            )
        )

        if protocol == "a2a":
            from any_agent.serving import A2AServingConfig
            serving_config = A2AServingConfig(port=5001, endpoint="/google")
        elif protocol == "mcp":
            from any_agent.serving import MCPServingConfig
            serving_config = MCPServingConfig(port=5001, endpoint="/google")

        server_handle = await agent.serve_async(serving_config)
        await server_handle.task

    if __name__ == "__main__":
        parser=argparse.ArgumentParser()
        parser.add_argument("protocol", choices=["a2a", "mcp"])
        args = parser.parse_args()
        asyncio.run(serve_agent(args.protocol))

    ```

=== "OpenAI Expert"

    ```python
    # openai_expert.py
    import argparse
    import asyncio

    from any_agent import AgentConfig, AnyAgent
    from any_agent.tools import search_web


    async def serve_agent(protocol):
        agent = await AnyAgent.create_async(
            "openai",
            AgentConfig(
                name="openai_expert",
                model_id="mistral/mistral-small-latest",
                instructions="You can answer questions about the OpenAI Agents SDK but nothing else.",
                description="An agent that can answer questions specifically and only about the OpenAI Agents SDK.",
                tools=[search_web]
            )
        )

        if protocol == "a2a":
            from any_agent.serving import A2AServingConfig
            serving_config = A2AServingConfig(port=5002, endpoint="/openai")
        elif protocol == "mcp":
            from any_agent.serving import MCPServingConfig
            serving_config = MCPServingConfig(port=5002, endpoint="/openai")

        server_handle = await agent.serve_async(serving_config)
        await server_handle.task

    if __name__ == "__main__":
        parser=argparse.ArgumentParser()
        parser.add_argument("protocol", choices=["a2a", "mcp"])
        args = parser.parse_args()
        asyncio.run(serve_agent(args.protocol))

    ```

We can now run each of the scripts in a separate terminal and leave them running in the background:

```bash
python google_expert.py mcp  ## or a2a
```

```bash
python openai_expert.py a2a  ## or mcp
```

## Using the served agents

Once the agents are being served using the chosen protocol, you can directly use them using the official clients
of each protocol:

- [A2A Client](https://a2a-protocol.org/latest/tutorials/python/6-interact-with-server/#understanding-the-client-code)
- [MCP Client](https://modelcontextprotocol.io/quickstart/client)

Alternatively, as described in [Using Agents-As-Tools](./agents/tools.md#using-agents-as-tools), we can run another python script containing the main agent that can use the served agents:

```python
import asyncio

from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPSse
from any_agent.tools import a2a_tool_async


async def main():
    prompt = "What do you know about the Google ADK?"

    google_expert = MCPSse(
        url="http://localhost:5001/google/sse", client_session_timeout_seconds=300)
    openai_expert = await a2a_tool_async(
        url="http://localhost:5002/openai")

    main_agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id="gemini/gemini-2.5-pro",
            instructions="You must use the available tools to answer.",
            tools=[google_expert, openai_expert],
        )
    )

    await main_agent.run_async(prompt)

if __name__ == "__main__":
    asyncio.run(main())

```

## Bridging MCP to A2A

any-agent provides a bridge that allows you to expose MCP servers as A2A-compatible services. This enables:

- **Protocol Interoperability**: Make MCP tools available to Google's A2A ecosystem
- **Enterprise Integration**: Connect local tools with enterprise agent systems
- **Identity Support**: Leverage AGNTCY Identity for secure tool access

### Basic Bridge Example

```python
import asyncio
from any_agent.config import MCPStdio
from any_agent.serving import (
    MCPToA2ABridgeConfig,
    serve_mcp_as_a2a_async,
)

async def bridge_mcp_server():
    # Configure MCP server
    mcp_config = MCPStdio(
        command="uvx",
        args=["mcp-server-time"],
        tools=["get_current_time"],
    )
    
    # Configure bridge
    bridge_config = MCPToA2ABridgeConfig(
        mcp_config=mcp_config,
        port=8080,
        endpoint="/time-bridge",
        server_name="time-server",
    )
    
    # Start the bridge
    bridge_handle = await serve_mcp_as_a2a_async(mcp_config, bridge_config)
    print(f"MCP server bridged to A2A at: http://localhost:8080/time-bridge")
    
    # Keep running
    await bridge_handle.task

if __name__ == "__main__":
    asyncio.run(bridge_mcp_server())
```

### Using the Bridged MCP Tool

Once bridged, the MCP tool can be accessed via A2A protocol:

```python
from any_agent import AgentConfig, AnyAgent
from any_agent.tools import a2a_tool_async

async def use_bridged_tool():
    # Create A2A tool from bridged MCP server
    time_tool = await a2a_tool_async(
        "http://localhost:8080/time-bridge",
        toolname="get_time_via_bridge",
    )
    
    # Use in an agent
    agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            model_id="mistral/mistral-small-latest",
            instructions="Use tools to answer questions.",
            tools=[time_tool],
        ),
    )
    
    result = await agent.run_async("What time is it?")
    print(result.final_output)
```

## More Examples

Check out our cookbook examples:

👉 [Serve an Agent with A2A (Jupyter Notebook)](./cookbook/serve_a2a.ipynb)

👉 [Use an A2A Agent as a tool (Jupyter Notebook)](./cookbook/a2a_as_tool.ipynb)

👉 [Bridge MCP servers to A2A (Jupyter Notebook)](./cookbook/mcp_a2a_bridge.ipynb)
