# Defining and Running Agents

## Defining Agents

To define any agent system you will always use the same imports:

```python
from any_agent import AgentConfig, AnyAgent, AgentRunError
# In these examples, the built-in tools will be used
from any_agent.tools import search_web, visit_webpage
```

Check [`AgentConfig`][any_agent.config.AgentConfig] for more info on how to configure agents.

### Single Agent

```python
agent = AnyAgent.create(
    "openai",  # See other options under `Frameworks`
    AgentConfig(
        model_id="mistral/mistral-small-latest",
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

Multi-Agent systems can be implemented [using Agent-As-Tools](./tools.md#using-agents-as-tools).

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
        model_id="mistral/mistral-small-latest",
        instructions="Check if the output contains any math",
        agent_args={
            "output_guardrails": [math_guardrail]
        }
    )
)
```

## Running Agents

```python
try:
    agent_trace = agent.run("Which Agent Framework is the best??")
    print(agent_trace.final_output)
except AgentRunError as are:
    agent_trace = are.trace
```

Check [`AgentTrace`][any_agent.tracing.agent_trace.AgentTrace] for more info on the return type.

Exceptions are wrapped in an [`AgentRunError`][any_agent.AgentRunError], that carries the original exception in the `__cause__` attribute. Additionally, its `trace` property holds the trace containing the spans collected so far.

### Async

If you are running in `async` context, you should use the equivalent [`create_async`][any_agent.AnyAgent.create_async] and [`run_async`][any_agent.AnyAgent.run_async] methods:

```python
import asyncio

async def main():
    agent = await AnyAgent.create_async(
        "openai",
        AgentConfig(
            model_id="mistral/mistral-small-latest",
            instructions="Use the tools to find an answer",
            tools=[search_web, visit_webpage]
        )
    )

    agent_trace = await agent.run_async("Which Agent Framework is the best??")
    print(agent_trace.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### Batch Processing

While any-agent doesn't provide a dedicated `.run_batch()` API, we recommend using `asyncio.gather` with the [`AnyAgent.run_async`][any_agent.AnyAgent.run_async] API for concurrent processing:

```python
import asyncio
from any_agent import AgentConfig, AnyAgent

async def process_batch():
    agent = await AnyAgent.create_async("tinyagent", AgentConfig(...))
    inputs = ["Input 1", "Input 2", "Input 3"]
    tasks = [agent.run_async(input_text) for input_text in inputs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### Multi-Turn Conversations

For scenarios where you need to maintain conversation history across multiple agent interactions, you can leverage the [`spans_to_messages`][any_agent.tracing.agent_trace.AgentTrace.spans_to_messages] method built into the AgentTrace. This function converts agent traces into a standardized message format that can be used to provide context in subsequent conversations.


!!! tip "When to Use Each Approach"

    - **Multi-turn with `spans_to_messages`**: When you need to maintain context across separate agent invocations or implement complex conversation management logic
    - **User interaction tools**: When you want the agent to naturally interact with users during its execution, asking questions as needed to complete its task
    - **Hybrid approach**: Combine both patterns for sophisticated agents that maintain long-term context while also gathering real-time user input


#### Basic Multi-Turn Example

```python
from any_agent import AgentConfig, AnyAgent

# Create your agent
agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="mistral/mistral-small-latest",
        instructions="You are a helpful assistant. Use previous conversation context when available.",
    )
)

response1 = agent.run("What's the capital of California?")
print(f"Agent: {response1.final_output}")
conversation_history = response1.spans_to_messages()
# Convert previous conversation to readable format
history_text = "\n".join([
    f"{msg.role.capitalize()}: {msg.content}"
    for msg in conversation_history
    if msg.role != "system"
])

user_message = "What's the closest national park to that city"

full_prompt = f"""Previous conversation:
{history_text}

Current user message: {user_message}

Please respond taking into account the conversation history above."""

response2 = agent.run(full_prompt)
print(f"Agent: {response2.final_output}")  # Agent will understand "that city" refers to Sacramento
```

#### Design Philosophy: Thoughtful Message History Management

You may notice that the `agent.run()` method doesn't accept a `messages` parameter directly. This is an intentional design choice to encourage thoughtful handling of conversation history by developers. Rather than automatically managing message history, any-agent empowers you to:

- **Choose your context strategy**: Decide what parts of conversation history are relevant
- **Manage token usage**: Control how much context you include to optimize costs and performance
- **Handle complex scenarios**: Implement custom logic for conversation branching, summarization, or context windowing

This approach ensures that conversation context is handled intentionally rather than automatically, leading to more efficient and purposeful agent interactions.

#### Using User Interaction Tools for Regular Conversations

For scenarios where you need regular, back-and-forth interaction with users, we recommend using or building your own **user interaction tools** rather than managing conversation history manually. This pattern allows the agent to naturally ask follow-up questions and gather information as needed. We provide a default `send_console_message` tool which uses console inputs and outputs, but you may need to use a more advanced tool (such as a Slack MCP Server) to handle user interaction.

```python
from any_agent import AgentConfig, AnyAgent
from any_agent.tools.user_interaction import send_console_message

# Create agent with user interaction capabilities
agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="mistral/mistral-small-latest",
        instructions="You are a helpful travel assistant. Send console messages to ask more questions. Do not stop until you've answered the question.",
        tools=[send_console_message]
    )
)

# The agent can now naturally ask questions during its execution
prompt = """
I'm planning a trip and need help finding accommodations.
Please ask me some questions to understand my preferences, then provide recommendations.
"""

agent_trace = agent.run(prompt)
print(f"Final recommendations: {agent_trace.final_output}")
```

This approach is demonstrated in our [MCP Agent cookbook example](../cookbook/mcp_agent.ipynb), where an agent uses user interaction tools to gather trip planning information dynamically. The agent can ask clarifying questions, get user preferences, and provide personalized recommendations all within a single `run()` call.
