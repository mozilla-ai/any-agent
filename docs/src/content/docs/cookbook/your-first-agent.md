---
description: Build and run your first agent with any-agent using simple web tools.
---

# Your First Agent

If you're looking to build your first agent using a few simple tools, this is a great place to start. In this cookbook example, we will create and run a simple agent that has access to a few web tools. This can be easily expanded to add more advanced tools and features.

[![Your first agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mozilla-ai/any-agent/blob/main/docs/cookbook/your_first_agent.ipynb)

## Install Dependencies

any-agent uses the python asyncio module to support async functionality. When running in Jupyter notebooks, this means we need to enable the use of nested event loops. We'll install any-agent and enable this below using nest_asyncio.

```python
%pip install 'any-agent' --quiet
%pip install ddgs --quiet

import warnings

import nest_asyncio

# Suppress technical warnings to reduce noise for the user
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

nest_asyncio.apply()
```

## Configure the Agent

### Pick an LLM

For this tutorial, we'll use Mistral's mistral-small-latest (fast and affordable).
You could also use:
- `gpt-4o-mini`
- `claude-3-5-sonnet-latest`
- Any other model supported by any-agent

### Pick Tools

We'll use `search_web`(DuckDuckGo), which provides Duck Duck Go search for free with no API key required.

```python
import os
from getpass import getpass

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass("Enter your Mistral API Key: ")
```

```python
from any_agent import AgentConfig, AnyAgent
from any_agent.tools import search_web

# We use the 'mistral-small-latest' model we promised in the text
agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(model_id="mistral:mistral-small-latest", tools=[search_web]),
)
```

## Run the Agent

Now we've configured our agent, so it's time to run it! Let's give it a simple task: find 5 trending new TV shows that were released recently.

```python
agent_trace = agent.run(
    "What are 5 tv shows that are trending in 2025? Check a few sites, and provide the name of the show, the exact release date, the genre, and a brief description of the show."
)
```

## View the results

The `agent.run` method returns an AgentTrace object, which has a few convenient attributes for displaying some interesting information about the run.

```python
print(agent_trace.final_output)  # Final answer
print(f"Duration: {agent_trace.duration.total_seconds():.2f} seconds")
print(f"Usage: {agent_trace.tokens.total_tokens:,}")
print(f"Cost (USD): {agent_trace.cost.total_cost:.6f}")
```
