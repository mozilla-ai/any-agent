# Creating an agent with MCP

The [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) (MCP) introduced by Anthropic has proven to be a popular method for providing an AI agent with access to a variety of tools. [This Huggingface blog post ](https://huggingface.co/blog/Kseniase/mcp) has a nice explanation of MCP.  In this tutorial, we'll build an agent that is able to leverage MCP server provided tools.

Note: because this tutorial relies upon advanced stdio/stderr communication using the MCP Server, it cannot be run on Google Colab.

## Install Dependencies

any-agent uses the python asyncio module to support async functionality. When running in Jupyter notebooks, this means we need to enable the use of nested event loops. We'll install any-agent and enable this below using nest_asyncio.

```python
%pip install 'any-agent' 'mcp-server-time' --quiet

import nest_asyncio

nest_asyncio.apply()
```

## Configure the Agent

Now it's time to configure the agent! At this stage you have a few choices:

### Pick the framework

We support a variety of underlying agent frameworks (OpenAI, Smolagents, Langchain, TinyAgent, etc), which all have their own particular agentic AI implementations. For this tutorial's simple use case, any of the frameworks should work just fine, but any-agent makes it easy to try out a different framework later, if we so choose. For this example, we will use the [TinyAgent](../agents/frameworks/tinyagent.md) framework.

### Pick an LLM

Regardless of which agent framework you choose, each framework supports any-llm, which is a proxy that allows us to use whichever LLM inside the framework, hosted by any provider. For example, we could use a local model via llama.cpp or llamafile, a Google-hosted gGemini model, or a AWS bedrock hosted Llama model. For this example, let's use Mistral AI's mistral:mistral-small-latest.

### Pick which tools to use

 In this example, we'll add a few MCP servers that we host locally, which means we'll use a Stdio MCP server. If an MCP Server is already running and hosted elsewhere, you can use an SSE connection to access it. You can browse some of the officially supported MCP servers [here](https://github.com/modelcontextprotocol/servers/tree/main?tab=readme-ov-file).

 Let's use two MCP servers:

 * [Time](https://github.com/modelcontextprotocol/servers/tree/main/src/time): so the agent can know what time/day it is.
 * [Airbnb](https://github.com/openbnb-org/mcp-server-airbnb): so the agent can browse airbnb listings (Needs [npx](https://docs.npmjs.com/cli/v8/commands/npx))

 We will also add a custom send_message tool, that way it can ask us additional questions before getting its final answer!

```python
import os
import shutil
from getpass import getpass

if "MISTRAL_API_KEY" not in os.environ:
    print("MISTRAL_API_KEY not found in environment!")
    api_key = getpass("Please enter your MISTRAL_API_KEY: ")
    os.environ["MISTRAL_API_KEY"] = api_key
    print("MISTRAL_API_KEY set for this session!")
else:
    print("MISTRAL_API_KEY found in environment.")

# Quick Environment Check (Airbnb tool requires npx/Node.js)
if not shutil.which("npx"):
    print(
        "⚠️ Warning: 'npx' was not found in your path. The Airbnb tool requires Node.js/npm to run."
    )
```

```python
import sys

from any_agent import AgentConfig, AnyAgent
from any_agent.config import MCPStdio

time_tool = MCPStdio(
    command=sys.executable,
    args=["-u", "-m", "mcp_server_time", "--local-timezone", "America/New_York"],
    tools=[
        "get_current_time",
    ],
    client_session_timeout_seconds=30,
)

print("Done init time_tool")
# This MCP tool relies upon npx https://docs.npmjs.com/cli/v8/commands/npx which comes standard with npm
airbnb_tool = MCPStdio(
    command="npx",
    args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
    client_session_timeout_seconds=30,
)
print("Done init airbnb_tool")
```

```python
# This is a custom tool that we will provide to the agent. For the agent to use the tool, we must provide a docstring
# and also have proper python typing for input and output parameters
def send_message(message: str) -> str:
    """Display a message to the user and wait for their response.

    Args:
        message: str
            The message to be displayed to the user.

    Returns:
        str: The response from the user.

    """
    if os.environ.get("IN_PYTEST") == "1":
        return "2 people, next weekend, low budget. Do not ask for any more information or confirmation."
    return input(message + " ")
```

```python
print("Start creating agent")
try:
    agent = await AnyAgent.create_async(
        "tinyagent",  # See all options in https://mozilla-ai.github.io/any-agent/
        AgentConfig(
            model_id="mistral:mistral-large-latest",
            tools=[time_tool, airbnb_tool, send_message],
        ),
    )
except Exception as e:
    print(f"❌ Failed to create agent: {e}")
print("Done creating agent")
```

## Run the Agent

Now we've configured our agent, so it's time to run it! Since it has access to airbnb listings as well as the current time, it's a perfect fit for helping me find a nice airbnb for the weekend.

```python
prompt = """
I am planning a trip to New York, NY for next weekend. Please act as my travel planner.

Follow these steps in strict order:
1. **Time Check:** Use the time tool to identify the specific dates for "next weekend."
2. **Information Gathering:** Ask me about my budget and number of guests. Do NOT search yet.
3. **Wait:** Wait for my response.
4. **Search:** ONLY after I reply, search for listings.
   **CRITICAL:** You MUST set the `location` parameter explicitly to "New York, NY" in the tool call.
"""

agent_trace = await agent.run_async(prompt)
```

## View the results

The `agent.run` method returns an AgentTrace object, which has a few convenient attributes for displaying some interesting information about the run.

```python
print(agent_trace.final_output)  # Final answer
print(f"Duration: {agent_trace.duration.total_seconds():.2f} seconds")
print(f"Total Tokens: {agent_trace.tokens.total_tokens:,}")
print(f"Total Cost (USD): {agent_trace.cost.total_cost:.6f}")
```
