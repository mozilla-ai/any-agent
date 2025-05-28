# Google Agent Development Kit (ADK)

[https://github.com/google/adk-python](https://github.com/google/adk-python)

## Default Agent Type

We use [`google.adk.agents.llm_agent.LlmAgent`](https://google.github.io/adk-docs/agents/llm-agents/) as default.
Check the reference to find additional supported `agent_args`.

## Default Model Type

We use [`google.adk.models.lite_llm.LiteLLM`](https://google.github.io/adk-docs/agents/models/#using-cloud-proprietary-models-via-litellm) as default.
Check the reference to find additional supported `model_args`.

## Run args

Check [`RunConfig`](https://google.github.io/adk-docs/runtime/runconfig/) to find additional supported `AnyAgent.run` args.

## Examples

### Limiting the number of steps

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.tools import search_web, visit_webpage
from google.adk.agents.run_config import RunConfig

agent = AnyAgent.create(
    "google",
    AgentConfig(
        model_id="gpt-4.1-nano",
        instructions="You must use the available tools to find an answer",
        tools=[search_web, visit_webpage]
    )
)

agent.run(
    "Which Agent Framework is the best??",
    run_config=RunConfig(
        max_llm_calls=3
    )
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
    "google",
    main_agent,
    managed_agents=managed_agents,
)
```
