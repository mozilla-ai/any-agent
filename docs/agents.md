# Agents

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

```python
agent = AnyAgent.create(
    "openai",  # See other options under `Frameworks`
    AgentConfig(
        model_id="gpt-4.1-mini",
        instructions="You are the main agent. Use the other available agents to find an answer",
    ),
    managed_agents=[
        AgentConfig(
            name="search_web_agent",
            description="An agent that can search the web",
            model_id="gpt-4.1-nano",
            tools=[search_web]
        ),
        AgentConfig(
            name="visit_webpage_agent",
            description="An agent that can visit webpages",
            model_id="gpt-4.1-nano",
            tools=[visit_webpage]
        )
    ]
)
```

## Running Agents

Regardless of the definition (single-agent or multi-agent), you can run the
agent as follows:

```python
agent_trace = agent.run("Which Agent Framework is the best??")
print(agent_trace.final_output)
```

Check [`AgentTrace`][any_agent.tracing.trace.AgentTrace] for more info on the return type.

### Async

If you are running in `async` context, you should use the equivalent [`create_async`][any_agent.AnyAgent.create_async] and [`run_async`][any_agent.AnyAgent.run_async] methods:

```python
import asyncio

async def main():
    agent = await AnyAgent.create_async(
        "openai",
        AgentConfig(
            model_id="gpt-4.1-mini",
            instructions="You are the main agent. Use the other available agents to find an answer",
        ),
        managed_agents=[
            AgentConfig(
                name="search_web_agent",
                description="An agent that can search the web",
                model_id="gpt-4.1-nano",
                tools=[search_web]
            ),
            AgentConfig(
                name="visit_webpage_agent",
                description="An agent that can visit webpages",
                model_id="gpt-4.1-nano",
                tools=[visit_webpage]
            )
        ]
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
