# smolagents

[https://github.com/huggingface/smolagents](https://github.com/huggingface/smolagents)

## Default Agent Type

We use [`smolagents.CodeAgent`](https://huggingface.co/docs/smolagents/reference/agents#smolagents.CodeAgent) as default.
Check the reference to find additional supported `agent_args`.

## Default Model Type

We use [`smolagents.LiteLLMModel`](https://huggingface.co/docs/smolagents/reference/models#smolagents.LiteLLMModel) as default.
Check the reference to find additional supported `model_args`.

## Run args

Check [`smolagents.MultiStepAgent.run`](https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.MultiStepAgent.run) to find additional supported `AnyAgent.run` args.

## Examples

### Limiting the number of steps

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "smolagents",
    AgentConfig(
        model_id="gpt-4.1-nano",
        instructions="You must use the available tools to find an answer",
        tools=[search_web, visit_webpage]
    )
)

agent.run(
    "Which Agent Framework is the best??",
    max_steps=3
)
```
