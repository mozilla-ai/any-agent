# Agents

## Defining Agents

To define any agent system you will always use the same imports:

```python
from any_agent import AgentConfig, AnyAgent, TracingConfig
# In these examples, the built-in tools will be used
from any_agent.tools import search_web, visit_webpage
```


### Model ID

`model_id` allows to select the underlying model used by the agent.
If you are using the default `model_type` (LiteLLM), you can refer to [LiteLLM Provider Docs](https://docs.litellm.ai/docs/providers) for the list of providers and how to access them.

!!! note

    If you plan on using a model that requires access to an external service (e.g. OpenAI, Mistral, DeepSeek, etc), you'll need to set any relevant environment variables, e.g.

    ```bash
    export OPENAI_API_KEY=your_api_key_here
    export DEEPSEEK_API_KEY=your_api_key_here
    ```

### Single Agent

```python
agent = AnyAgent.create(
    "openai",  # See other options under `Frameworks`
    AgentConfig(
        model_id="gpt-4.1-nano",
        instructions="Use the tools to find an answer",
        tools=[search_web, visit_webpage]
    ),
    tracing=TracingConfig(output_dir="traces") # Optional, but recommended for saving and viewing traces
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
agent.run("Which Agent Framework is the best??")
```

### Async

If you are running in `async` context, you should use the equivalent `create_async` and `run_async` methods:

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
        ],
        tracing=TracingConfig()
    )

    await agent.run_async("Which Agent Framework is the best??")


if __name__ == "__main__":
    asyncio.run(main())

```

## Advanced configuration

!!! tip

    Check the `Frameworks` pages for more details on each of these
    configuration options.

### Agent Args

`agent_args` are passed when creating the instance used by the underlying framework.

For example, you can pass `output_type` when using the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python):

```python
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = AnyAgent.create(
    AgentConfig(
        model_id="gpt-4.1-mini",
        instructions="Extract calendar events from text",
        agent_args={
            "output_type": CalendarEvent
        }
    )
)
```

### Agent Type

`agent_type` controls the type of agent class that is used by the framework, and is unique to the framework used.

Check the individual `Frameworks` pages for more info on the defaults.

### Model Args

`model_args` allows to set parameters like `temperature`, `top_k`, as well as any other provider-specific parameters.
Refer to [LiteLLM Completion API Docs](https://docs.litellm.ai/docs/text_completion) for more info.

### Model Type

`model_type` controls the type of model class that is used by the agent framework, and is unique to the agent framework being used.

For each framework, we leverage their support for [`LiteLLM`](https://github.com/BerriAI/litellm) and use it as default `model_type`, allowing you to use the same `model_id` syntax across these frameworks.

### Run Args

You can pass arbitrary `key=value` arguments to `agent.run` and they will be forwarded
to the corresponding method used by the underlying framework.

For example you can pass `max_turns=30` when using the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python):

```python
agent.run("Which agent framework is the best?", max_turns=30)
```


### Cleaning up the Agent

When an AnyAgent object is deleted, the python garbage collector cleans up any resources owned by the object. However, when running or re-creating an agent in the same python process (for example, in test scripts) it may be necessary to forcefully shut down the agent to avoid unexpected side affects. For this purpose, `agent.exit` is available which will shut down all resources the agent was using.

For example,

```python
agent.run("Which agent framework is the best?")
agent.exit() # cleans up the agent synchronously
```
