# **any-agent**

`any-agent` is a Python library providing a single interface to different agent frameworks.

!!! warning

    Compared to traditional code-defined workflows, agent frameworks introduce complexity and
    demand much more computational power.

    Before jumping to use one, carefully consider and evaluate how much value you
    would get compared to manually defining a sequence of tools and LLM calls.

## Quickstart

```bash
pip install any-agent
```

To define any agent system you will always use the same imports:

```python
from any_agent import AgentConfig, AgentFramework, AnyAgent
```

!!! note

    If you plan on using an agent that requires access to an external service (e.g. OpenAI, Mistral, DeepSeek, etc), you'll need to set any relevant environment variables, e.g.

    ```bash
    export OPENAI_API_KEY=your_api_key_here
    export DEEPSEEK_API_KEY=your_api_key_here
    ```

### Single Agent

Configure the agent:

```python
from any_agent.tools import search_web, visit_webpage
main_agent = AgentConfig(
    model_id="gpt-4o",
    tools=[search_web, visit_webpage]
)
```

Choose one of the available frameworks:

```python
from random import choice

framework = AgentFramework(
    choice(
        ["langchain", "llama_index", "openai", "smolagents"]
    )
)
```

Create and run the agent:

```python
agent = AnyAgent.create(framework, main_agent)

agent.run("Which Agent Framework is the best??")
```

### Multi-Agent

Building on top of the previous example, we can easily extend it to a multi-agent system.

!!! warning

    A multi-agent system introduces even more complexity than a single agent.

    As stated before, carefully consider whether you need to adopt this pattern to
    solve the task.

First, configure the `main_agent`, similar to before:

```python
main_agent = AgentConfig(
    model_id="gpt-4o",
    description="Main Agent"
)
```

This agent will act as the "orchestrator".

Then, configure the list of `managed_agents`:

```python
from any_agent.tools import search_web, visit_webpage
managed_agents = [
    AgentConfig(
        name="search_web_agent",
        model_id="gpt-4o-mini",
        description="Agent that can search the web",
        tools=[search_web]
    ),
    AgentConfig(
        name="visit_webpage_agent",
        model_id="gpt-4o-mini",
        description="Agent that can visit webpages",
        tools=[visit_webpage]
    )
]
```

You can then create and run the multi-agent:

```python
multi_agent = AnyAgent.create(framework, main_agent, managed_agents)

multi_agent.run("Which Agent Framework is the best??")
```
