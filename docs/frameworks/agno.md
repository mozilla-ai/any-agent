# Agno

[https://github.com/agno-agi/agno](https://github.com/agno-agi/agno)

## Default Agent Type

We use [`agno.agent.Agent`](https://docs.agno.com/reference/agents/agent) as default.
Check the reference to find additional supported `agent_args`.

## Default Model Type

We use [`agno.models.litellm.LiteLLM`](https://docs.agno.com/models/litellm) as default.
Check the reference to find additional supported `model_args`.

## Examples

### Limiting the number of steps

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "agno",
    AgentConfig(
        model_id="gpt-4.1-nano",
        instructions="You must use the available tools to find an answer",
        tools=[search_web, visit_webpage]
    ),
    agent_args={
        "tool_call_limit": 3
    }
)
agent.run("Which Agent Framework is the best??")
```
