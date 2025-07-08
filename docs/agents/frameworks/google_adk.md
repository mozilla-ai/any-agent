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
        model_id="mistral/mistral-small-latest",
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
