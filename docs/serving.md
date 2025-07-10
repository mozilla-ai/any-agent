# Serving

`any-agent` provides a simple way of serving agents from any of the supported frameworks using the
[Agent2Agent Protocol (A2A)](https://google.github.io/A2A/), via the [A2A Python SDK](https://github.com/google-a2a/a2a-python), or using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/specification/2025-03-26), via the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk). You can refer to the links for more information on
these protocols, as explaining them is out of the scope of this page.

!!! warning

    The A2A protocol is in early stages of development and so is the functionality provided by `any-agent` here.

In order to use A2A serving, you must first install the 'a2a' extra: `pip install 'any-agent[a2a]'`

You can configure and serve an agent using the [`A2AServingConfig`][any_agent.serving.A2AServingConfig] and the [`AnyAgent.serve_async`][any_agent.AnyAgent.serve_async] method.

## Running Async Servers in Sync Environments

Since `any-agent` uses async/await patterns for better performance and resource management, the serving functions are async by default. However, you can easily run async servers in sync environments using Python's `asyncio` utilities:

### Using `asyncio.run()`

The simplest approach is to wrap your async code in `asyncio.run()`:

```python
import asyncio
from any_agent import AgentConfig, AnyAgent
from any_agent.serving import A2AServingConfig

async def main():
    agent = await AnyAgent.create_async(
        "tinyagent",
        AgentConfig(
            name="my_agent",
            model_id="mistral/mistral-small-latest",
            description="A helpful agent"
        )
    )

    server_handle = await agent.serve_async(A2AServingConfig(port=8080))

    try:
        # Keep the server running
        await server_handle.task
    except KeyboardInterrupt:
        await server_handle.shutdown()

asyncio.run(main())
```

## Serving via A2A

### Example

For illustrative purposes, we are going to define 2 separate scripts, each defining an agent to answer questions about a specific agent framework (either OpenAI Agents SDK or Google ADK):


=== "Google Expert"

    ```python
    # google_expert.py
    import asyncio
    from any_agent import AgentConfig, AnyAgent
    from any_agent.serving import A2AServingConfig
    from any_agent.tools import search_web

    async def main():
        agent = await AnyAgent.create_async(
            "google",
            AgentConfig(
                name="google_expert",
                model_id="mistral/mistral-small-latest",
                description="An agent that can answer questions specifically and only about the Google Agents Development Kit (ADK). Reject questions about anything else.",
                tools=[search_web]
            )
        )

        server_handle = await agent.serve_async(A2AServingConfig(port=5001))
        await server_handle.task

    asyncio.run(main())
    ```

=== "OpenAI Expert"

    ```python
    # openai_expert.py
    import asyncio
    from any_agent import AgentConfig, AnyAgent
    from any_agent.serving import A2AServingConfig
    from any_agent.tools import search_web

    async def main():
        agent = await AnyAgent.create_async(
            "openai",
            AgentConfig(
                name="openai_expert",
                model_id="mistral/mistral-small-latest",
                instructions="You can answer questions about the OpenAI Agents SDK but nothing else.",
                description="An agent that can answer questions specifically about the OpenAI Agents SDK.",
                tools=[search_web]
            )
        )

        server_handle = await agent.serve_async(A2AServingConfig(port=5002))
        await server_handle.task

    asyncio.run(main())
    ```

We can then run each of the scripts in a separate terminal and leave them running in the background.

Now, using a simple python script that implements the A2A client, we can communicate with these agents! For this example,
we use the [A2A Python SDK](https://github.com/google-a2a/a2a-python)


```python
from uuid import uuid4
import asyncio
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import AgentCard, MessageSendParams, SendMessageRequest

async def main():
    async with httpx.AsyncClient() as httpx_client:
        agent_card: AgentCard = await A2ACardResolver(
            httpx_client,
            base_url="http://localhost:5001",
        ).get_agent_card(http_kwargs=None)
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        send_message_payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "What do you know about the Google ADK?"}],
                "messageId": uuid4().hex,
            },
        }
        request = SendMessageRequest(params=MessageSendParams(**send_message_payload))
        response = await client.send_message(request, http_kwargs={"timeout": 60})
        print(f" Response from first agent: {response.model_dump_json(indent=2)}")

        agent_card: AgentCard = await A2ACardResolver(
            httpx_client,
            base_url="http://localhost:5002",
        ).get_agent_card(http_kwargs=None)
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        send_message_payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "What do you know about the Google ADK?"}],
                "messageId": uuid4().hex,
            },
        }
        request = SendMessageRequest(params=MessageSendParams(**send_message_payload))
        response = await client.send_message(request, http_kwargs={"timeout": 60})
        print(f" Response from second agent: {response.model_dump_json(indent=2)}")
if __name__ == "__main__":
    asyncio.run(main())
```

You will see that the first agent answered the question, but the second agent did not answer the question.
This is because the question was about Google ADK,
but the agent was told it could only answer questions about the OpenAI Agents SDK.

### Advanced Configuration

#### Custom Skills

By default, an agent's skills are automatically inferred from its tools. However, you can explicitly define skills for more control over the agent card:

```python
from a2a.types import AgentSkill
from any_agent.serving import A2AServingConfig

# Define custom skills
custom_skills = [
    AgentSkill(
        id="web-search",
        name="search_web",
        description="Search the web for current information",
        tags=["search", "web", "information"]
    ),
    AgentSkill(
        id="data-analysis",
        name="analyze_data",
        description="Analyze datasets and provide insights",
        tags=["analysis", "data", "insights"]
    )
]

config = A2AServingConfig(
    port=8080,
    skills=custom_skills
)
```

### More Examples

Check out our cookbook example for building and serving an agent via A2A:

👉 [Serve an Agent with A2A (Jupyter Notebook)](./cookbook/serve_a2a.ipynb)

### Accessing an A2A agent using tools

As described in the [tools section](./agents/tools.md#a2a-tools), an agent can request actions from other agents by using the `a2a_tool`  or `a2a_tool_async` function. It retrieves the agent card, and builds another function that relays the request via the A2A protocol and unpacks the result.

## Serving via MCP

### Example

In a similar way to [the A2A example](#example), we are going to define two agents served over MCP:

=== "Google Expert"

    ```python
    # google_expert.py
    import asyncio
    from any_agent import AgentConfig, AnyAgent
    from any_agent.serving import MCPServingConfig
    from any_agent.tools import search_web

    async def main():
        agent = await AnyAgent.create_async(
            "google",
            AgentConfig(
                name="google_expert",
                model_id="mistral/mistral-small-latest",
                description="An agent that can answer questions specifically and only about the Google Agents Development Kit (ADK). Reject questions about anything else.",
                tools=[search_web]
            )
        )

        server_handle = await agent.serve_async(MCPServingConfig(port=5001, endpoint="/google"))
        await server_handle.task

    asyncio.run(main())
    ```

=== "OpenAI Expert"

    ```python
    # openai_expert.py
    import asyncio
    from any_agent import AgentConfig, AnyAgent
    from any_agent.serving import MCPServingConfig
    from any_agent.tools import search_web

    async def main():
        agent = await AnyAgent.create_async(
            "openai",
            AgentConfig(
                name="openai_expert",
                model_id="mistral/mistral-small-latest",
                instructions="You can provide information about the OpenAI Agents SDK but nothing else (specially, nothing about the Google SDK).",
                description="An agent that can answer questions specifically about the OpenAI Agents SDK.",
                tools=[search_web]
            )
        )

        server_handle = await agent.serve_async(MCPServingConfig(port=5002, endpoint="/openai"))
        await server_handle.task

    asyncio.run(main())
    ```

We can then run each of the scripts in a separate terminal and leave them running in the background.

Then, we run another python script containing the main agent that will contact one of the other two via MCP:

```python
# Main agent

from uuid import uuid4
import asyncio
from any_agent.config import MCPSse
from any_agent import AgentConfig, AgentFramework, AnyAgent

async def main():
    prompt = "What do you know about the Google ADK?"
    google_server_url = f"http://localhost:5001/google/sse"
    openai_server_url = f"http://localhost:5002/openai/sse"

    main_agent_cfg = AgentConfig(
        instructions="Use the available tools to obtain additional information to answer the query.",
        description="The orchestrator that can use other agents via tools using the MCP protocol.",
        tools=[
            MCPSse(url=google_server_url, client_session_timeout_seconds=300),
            MCPSse(url=openai_server_url, client_session_timeout_seconds=300),
        ],
        model_id="mistral/mistral-small-latest",
    )

    main_agent = await AnyAgent.create_async(
        agent_framework=AgentFramework.OPENAI,
        agent_config=main_agent_cfg,
    )

    agent_trace = await main_agent.run_async(prompt)

    print(agent_trace)

if __name__ == "__main__":
    asyncio.run(main())
```
