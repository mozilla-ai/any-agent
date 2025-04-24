# **any-agent**

`any-agent` is a Python library providing a single interface to different agent frameworks.

!!! warning

    Compared to traditional code-defined workflows, agent frameworks introduce complexity and
    demand much more computational power.

    Before jumping to use one, carefully consider and evaluate how much value you
    would get compared to manually defining a sequence of tools and LLM calls.

## Requirements

- Python 3.11 or newer

## Quickstart

```bash
pip install any-agent
```

To define any agent system you will always use the same imports:

```python
from any_agent import AgentConfig, AnyAgent, TracingConfig
```

!!! note

    If you plan on using an agent that requires access to an external service (e.g. OpenAI, Mistral, DeepSeek, etc), you'll need to set any relevant environment variables, e.g.

    ```bash
    export OPENAI_API_KEY=your_api_key_here
    export DEEPSEEK_API_KEY=your_api_key_here
    ```

### Single Agent

```py
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "smolagents",  # See all options in https://mozilla-ai.github.io/any-agent/frameworks/
    AgentConfig(
        model_id="gpt-4.1-nano",
        instructions="Use the tools to find an answer",
        tools=[search_web, visit_webpage]
    )
    tracing=TracingConfig(output_dir="traces") # Optional, but recommended for saving and viewing traces
)

agent.run("Which Agent Framework is the best??")
```

### Multi-Agent

!!! warning

    A multi-agent system introduces even more complexity than a single agent.

    As stated before, carefully consider whether you need to adopt this pattern to
    solve the task.

```py
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "smolagents",  # See all options in https://mozilla-ai.github.io/any-agent/frameworks/
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

agent.run("Which Agent Framework is the best??")
```

## Async

If you are running in `async` context, you should use the equivalent `create_async` and `run_async` methods:

```py
import asyncio
from any_agent.tools import search_web, visit_webpage

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
        ],
        tracing=TracingConfig()
    )

    await agent.run_async("Which Agent Framework is the best??")

if __name__ == "__main__":
    asyncio.run(main())
```
