# OpenAI Agents SDK

[https://github.com/openai/openai-agents-python](https://github.com/openai/openai-agents-python)

## Default Agent Type

We use [`agents.Agent`](ttps://openai.github.io/openai-agents-python/ref/agent/#agents.agent.Agent) as default.
Check the reference to find additional supported `agent_args`.

## Default Model Type

We use [`agents.extensions.models.litellm_model.LitellmModel`](https://openai.github.io/openai-agents-python/ref/extensions/litellm/) as default.
Check the reference to find additional supported `model_args`.

## Run args

Check [`agents.run.Runner.run`](https://openai.github.io/openai-agents-python/ref/run/#agents.run.Runner.run) to find additional supported `AnyAgent.run` args.

## Examples

### Limiting the number of steps

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "openai",
    AgentConfig(
        model_id="gpt-4.1-nano",
        instructions="You must use the available tools to find an answer",
        tools=[search_web, visit_webpage]
    )
)

agent.run(
    "Which Agent Framework is the best??",
    max_turns=3
)
```

### Using `handoff`

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.tools import search_web, show_final_output, visit_webpage

main_agent = AgentConfig(
    model_id="o3-mini",
)

managed_agents = [
    AgentConfig(
        model_id="gpt-4o",
        name="search-web-agent",
        tools=[
            search_web,
            visit_webpage,
        ],
    ),
    AgentConfig(
        model_id="gpt-4o-mini",
        name="communication-agent",
        tools=[show_final_output],
        agent_args={
            "handoff": True
        }
    ),
]

AnyAgent.create(
    "openai",
    main_agent,
    managed_agents=managed_agents,
)
```
