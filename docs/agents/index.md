# Defining and Running Agents

## Defining Agents

To define any agent system you will always use the same imports:

```python
from any_agent import AgentConfig, AnyAgent
# In these examples, the built-in tools will be used
from any_agent.tools import search_web, visit_webpage
```

Check [`AgentConfig`][any_agent.config.AgentConfig] for more info on how to configure agents.

### Single Agent

```python
agent = AnyAgent.create(
    "openai",  # See other options under `Frameworks`
    AgentConfig(
        model_id="gpt-4.1-nano",
        instructions="Use the tools to find an answer",
        tools=[search_web, visit_webpage]
    ),
)
```

### Multi-Agent

!!! warning

    A multi-agent system introduces even more complexity than a single agent.

    As stated before, carefully consider whether you need to adopt this pattern to
    solve the task.

Multi-agent can be implemented today using the A2A protocol (see [A2A docs](https://mozilla-ai.github.io/any-agent/serving/)) and will be also supported with Agent-As-Tools (follow progress at https://github.com/mozilla-ai/any-agent/issues/382).

### Framework Specific Arguments

The `agent_args` parameter in `AgentConfig` allows you to pass arguments specific to the underlying framework that the agent instance is built on.

**Example-1**: To pass the `output_type` parameter for structured output, when using the OpenAI Agents SDK:

```python
from pydantic import BaseModel
from any_agent import AgentConfig, AgentFramework, AnyAgent

class BookInfo(BaseModel):
    title: str
    author: str
    publication_year: int

framework = AgentFramework.OPENAI

agent = AnyAgent.create(
    framework,
    AgentConfig(
        model_id="gpt-4.1-mini",
        instructions="Extract book information from text",
        agent_args={
            "output_type": BookInfo
        }
    )
)
```

**Example-2**: In smolagents, for structured output one needs to use the `grammar` parameter. Additionally, `planning_interval` defines the interval at which the agent will run a planning step.

```python
from pydantic import BaseModel
from any_agent import AgentConfig, AgentFramework, AnyAgent


framework = AgentFramework.SMOLAGENTS

class WebPageInfo(BaseModel):
    title: str
    summary: str

agent = AnyAgent.create(
    framework,
    AgentConfig(
        model_id="gpt-4.1-mini",
        instructions="Extract webpage title and summary from url",
        agent_args={
            "planning_interval": 1,
            "grammar": WebPageInfo
        }
    )
)
```

## Running Agents

```python
agent_trace = agent.run("Which Agent Framework is the best??")
print(agent_trace.final_output)
```

Check [`AgentTrace`][any_agent.tracing.agent_trace.AgentTrace] for more info on the return type.

### Async

If you are running in `async` context, you should use the equivalent [`create_async`][any_agent.AnyAgent.create_async] and [`run_async`][any_agent.AnyAgent.run_async] methods:

```python
import asyncio

async def main():
    agent = await AnyAgent.create_async(
        "openai",
        AgentConfig(
            model_id="gpt-4.1-mini",
            instructions="Use the tools to find an answer",
            tools=[search_web, visit_webpage]
        )
    )

    agent_trace = await agent.run_async("Which Agent Framework is the best??")
    print(agent_trace.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Cleaning up the Agent

When an AnyAgent object is deleted, the python garbage collector cleans up any resources owned by the object. However, when running or re-creating an agent in the same python process (for example, in test scripts) it may be necessary to forcefully shut down the agent to avoid unexpected side affects. For this purpose, `agent.exit` is available which will shut down all resources the agent was using.

For example,

```python
agent.run("Which agent framework is the best?")
agent.exit() # cleans up the agent synchronously
```
