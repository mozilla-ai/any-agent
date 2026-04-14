# Use an Agent as a tool for another agent.

Multi-Agent systems are complicated! Enter the [A2A](https://github.com/google-a2a/A2A) protocol by Google: this protocol allows for simpler communication between agents, and easily enables the use of an agent as a tool for another agent. In this tutorial we'll show how you can create a few agents with any-agent and have an agent be provided as a tool for the other agent.

This tutorial assumes basic familiarity with any-agent: if you haven't used any-agent before you may also find the [Creating your first agent](your-first-agent.md) cookbook to be useful. You also may find the other A2A related cookbook to be useful: [Serve an Agent with A2A](serve-a2a.md)

Note: because this tutorial relies upon advanced stdio/stderr communication using the MCP Server, it cannot be run on Google Colab.

## Install Dependencies

any-agent uses the python asyncio module to support async functionality. When running in Jupyter notebooks, this means we need to enable the use of nested event loops. We'll install any-agent and enable this below using nest_asyncio.

```python
%pip install 'any-agent[a2a]'

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

## Configure the first two agents and serve them over A2A

Let's give our first two "helper" agents some very simple capabilities. For this demo, we'll use the async method `agent.serve_async` so that we can easily run all of the agents from inside the notebook in a single python process.

```python
import asyncio

import httpx

from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStdio
from any_agent.serving import A2AServingConfig

# This MCP Tool relies upon uvx https://docs.astral.sh/uv/getting-started/installation/
time_tool = MCPStdio(
    command="uvx",
    args=["mcp-server-time", "--local-timezone=America/New_York"],
    tools=[
        "get_current_time",
    ],
)

time = await AnyAgent.create_async(
    "tinyagent",  # See all options in https://docs.mozilla.ai/any-agent/
    AgentConfig(
        model_id="mistral:mistral-small-latest",
        name="time_agent",
        description="I'm an agent to help with getting the time",
        tools=[time_tool],
    ),
)
time_handle = await time.serve_async(A2AServingConfig(port=0, endpoint="/time"))

weather_expert = await AnyAgent.create_async(
    "tinyagent",
    AgentConfig(
        model_id="mistral:mistral-small-latest",
        name="weather_expert",
        instructions="You're an expert that is an avid skier, recommend a great location to ski given a time of the year",
        description="I'm an agent that is an expert in recommending my favorite location given a time of the year",
    ),
)
weather_handle = await weather_expert.serve_async(
    A2AServingConfig(port=0, endpoint="/location_recommender")
)

weather_port = weather_handle.port
time_port = time_handle.port

max_attempts = 20
poll_interval = 0.5
attempts = 0
time_url = f"http://localhost:{time_port}"
weather_url = f"http://localhost:{weather_port}"
async with httpx.AsyncClient() as client:
    while True:
        try:
            # Try to make a basic GET request to check if server is responding
            await client.get(time_url, timeout=1.0)
            print(f"Server is ready at {weather_url}")
            break
        except (httpx.RequestError, httpx.TimeoutException):
            # Server not ready yet, continue polling
            pass

        await asyncio.sleep(poll_interval)
        attempts += 1
        if attempts >= max_attempts:
            msg = f"Could not connect to the servers. Tried {max_attempts} times with {poll_interval} second interval."
            raise ConnectionError(msg)
```

### Configure and use the Third Agent

Now that the first two agents are serving over A2A, our main agent can be given access to use it just like a tool! In order to do this you use the `a2a_tool_async` function which retrieves the info about the agent and allows you to call it.

```python
from any_agent import AgentConfig, AnyAgent
from any_agent.tools import a2a_tool_async

config = AgentConfig(
    model_id="mistral:mistral-small-latest",
    instructions="Use the available tools to obtain additional information to answer the query.",
    description="The orchestrator that can use other agents via tools using the A2A protocol.",
    tools=[
        await a2a_tool_async(
            f"http://localhost:{time_port}/time",
            http_kwargs={
                "timeout": 30
            },  # This gives the time agent up to 30 seconds to respond to each request
        ),
        await a2a_tool_async(
            f"http://localhost:{weather_port}/location_recommender",
            http_kwargs={
                "timeout": 30
            },  # This gives the weather agent up to 30 seconds to respond to each request
        ),
    ],
)

agent = await AnyAgent.create_async(
    agent_framework="tinyagent",
    agent_config=config,
)

agent_trace = await agent.run_async("Where should I go this weekend to ski?")

print(agent_trace.final_output)
```

### Shut down the servers and the agents

The following code can be used to gracefully shut down the agents that are serving via A2A.

```python
await time_handle.shutdown()
await weather_handle.shutdown()
```
