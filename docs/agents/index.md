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

Sometimes, there may be a new feature in a framework that you want to use that isn't yet supported universally in any-agent. The `agent_args` parameter in `AgentConfig` allows you to pass arguments specific to the underlying framework that the agent instance is built on.

**Example-1**: To pass the `output_guardrails` parameter, when using the OpenAI Agents SDK:

```python
from pydantic import BaseModel
from any_agent import AgentConfig, AgentFramework, AnyAgent
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
)

class MessageOutput(BaseModel):
    response: str

class MathOutput(BaseModel):
    reasoning: str
    is_math: bool

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any math.",
    output_type=MathOutput,
)

@output_guardrail
async def math_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math,
    )

framework = AgentFramework.OPENAI

agent = AnyAgent.create(
    framework,
    AgentConfig(
        model_id="gpt-4.1-mini",
        instructions="Check if the output contains any math",
        agent_args={
            "output_guardrails": [math_guardrail]
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
