# Serve an Agent with A2A

Once you've built an agent, a common question might be: "how do I let other developers/applications access it?". Enter the [A2A](https://github.com/google-a2a/A2A) protocol by Google! A2A is "An open protocol enabling communication and interoperability between opaque agentic applications". Any-Agent provides support for serving an agent over A2A, as simple as calling `await agent.serve_async()`. In this tutorial, we'll build and serve an agent using any-agent, and show how you can serve and interact with it via the A2A protocol.

![](/images/serve_a2a.png)

This tutorial assumes basic familiarity with any-agent: if you haven't used any-agent before you may also find the [Creating your first agent](./../your_first_agent) cookbook to be useful.

Note: because this tutorial relies upon advanced stdio/stderr communication using the MCP Server, it cannot be run on Google Colab.

## Install Dependencies

any-agent uses the python asyncio module to support async functionality. When running in Jupyter notebooks, this means we need to enable the use of nested event loops. We'll install any-agent and enable this below using nest_asyncio.

```python
%pip install 'any-agent[a2a]' 'mcp-server-time' --quiet

import nest_asyncio

nest_asyncio.apply()
```

```python
import os
from getpass import getpass

# This notebook communicates with Mistral models using the Mistral API.
if "MISTRAL_API_KEY" not in os.environ:
    print("MISTRAL_API_KEY not found in environment!")
    api_key = getpass("Please enter your MISTRAL_API_KEY: ")
    os.environ["MISTRAL_API_KEY"] = api_key
    print("MISTRAL_API_KEY set for this session!")
else:
    print("MISTRAL_API_KEY found in environment.")
```

## Configure and run the server

Let's give our agent the very simple capability to access the current time through a Model Context Protocol (MCP) server. For this demo, we'll use the async method `agent.serve_async` so that we can easily run both the server and client from inside the notebook.

```python
import asyncio
import sys

import httpx

from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStdio
from any_agent.serving import A2AServingConfig

time_tool = MCPStdio(
    command=sys.executable,
    args=["-u", "-m", "mcp_server_time", "--local-timezone", "America/New_York"],
    tools=[
        "get_current_time",
    ],
    client_session_timeout_seconds=30,
)

time = await AnyAgent.create_async(
    "tinyagent",  # See all options in https://mozilla-ai.github.io/any-agent/
    AgentConfig(
        model_id="mistral:mistral-small-latest",
        description="I'm an agent to help with getting the time",
        tools=[time_tool],
    ),
)

time_handle = await time.serve_async(A2AServingConfig(port=0))

server_port = time_handle.port

max_attempts = 20
poll_interval = 0.5
attempts = 0
server_url = f"http://localhost:{server_port}"
async with httpx.AsyncClient() as client:
    while True:
        try:
            # Try to make a basic GET request to check if server is responding
            await client.get(server_url, timeout=1.0)
            print(f"Server is ready at {server_url}")
            break
        except (httpx.RequestError, httpx.TimeoutException):
            # Server not ready yet, continue polling
            pass

        await asyncio.sleep(poll_interval)
        attempts += 1
        if attempts >= max_attempts:
            msg = f"Could not connect to {server_url}. Tried {max_attempts} times with {poll_interval} second interval."
            raise ConnectionError(msg)
```

## Call the agent using A2AClient

Now that the agent is listening on localhost, we can communicate with it from any other application that supports A2A. For this tutorial we'll use the a2a python SDK, but any client that implements the [A2A Protocol](https://github.com/google-a2a/A2A) would also work.

### A2A Agent Card

Before giving the agent a task, we first retrieve the agent's model card, which is a description of the agents capabilities. This helps the client understand what the agent can do, and what A2A features are available.

```python
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import AgentCard

# Create the httpx client
httpx_client = httpx.AsyncClient()

agent_card: AgentCard = await A2ACardResolver(
    httpx_client,
    base_url=f"http://localhost:{server_port}",
).get_agent_card(http_kwargs=None)
print(agent_card.model_dump_json(indent=2))

client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
```

### Make a request of the agent

Now that we've connected to the agent with the A2AClient, we're ready to make a request of the agent!

```python
from uuid import uuid4

from a2a.types import MessageSendParams, SendMessageRequest

send_message_payload = {
    "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "What time is it?"}],
        "messageId": uuid4().hex,
    },
}
request = SendMessageRequest(
    id=str(uuid4()), params=MessageSendParams(**send_message_payload)
)
response = await client.send_message(request, http_kwargs={"timeout": 30.0})
# Close the httpx client when done
await httpx_client.aclose()
```

```python
print(response.model_dump_json(indent=2))
```

## Cleanup

In order to shut down the A2A server, we can set the `should_exit` property of the server, which will cause the server to shutdown and the asyncio task to complete.

```python
await time_handle.shutdown()
print("Agent server has been shut down!")
```
