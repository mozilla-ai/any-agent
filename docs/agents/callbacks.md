# Agent Callbacks

Callbacks provide hooks into the lifecycle of an `AnyAgent` execution. Using callbacks, you can monitor, control, and extend agent behavior without modifying the core underlying agent logic.

## Implementing Callbacks

All callbacks must inherit from the base [`Callback`][any_agent.callbacks.base.Callback] class and can choose to implement any subset of the available callback methods. These methods include:

| Callback Method | When It Fires | Example Use Cases |
|:----------------:|:------------:|:----------------|
| before_agent_invocation | Once at start, before any LLM calls | Initialize counters, validate inputs, set up logging |
| before_llm_call | Before each LLM API call | Content filtering, cost tracking, prompt inspection |
| after_llm_call | After LLM responds, before adding to history | Response validation, token counting, logging |
| before_tool_execution | Before each tool runs | Rate limiting, input validation, authorization checks |
| after_tool_execution | After tool completes | Result validation, metrics collection, error handling |
| after_agent_invocation | Once at end, before returning final response | Cleanup, final metrics, audit logging |

```py
# Minimum valid implementation
def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
    return context  # <--- Essential!
```
## Managing State (`Context`)

During an agent run ( [`agent.run_async`][any_agent.AnyAgent.run_async] or [`agent.run`][any_agent.AnyAgent.run] ), a unique [`Context`][any_agent.callbacks.context.Context] object is created and shared across all callbacks.

Use `Context.shared` (a dictionary) to persist data across different steps and callbacks.

> Note: The `Context` object is mutable. You should modify `Context.shared` directly and return the same object.

`any-agent` populates the [`Context.current_span`][any_agent.callbacks.context.Context.current_span]
property so that callbacks can access information in a framework-agnostic way.

You can see what attributes are available for LLM Calls and Tool Executions by examining the [`GenAI`][any_agent.tracing.attributes.GenAI] class.

**Common Pattern**: Initialize a counter in one callback and check it in another.

```python
from any_agent.callbacks import Callback, Context
from any_agent.tracing.attributes import GenAI

class CountSearchWeb(Callback):
    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        if "search_web_count" not in context.shared:
            context.shared["search_web_count"] = 0
        if context.current_span.attributes[GenAI.TOOL_NAME] == "search_web":
            context.shared["search_web_count"] += 1
        return context
```

Callbacks can raise exceptions to stop agent execution. This is useful for implementing safety guardrails or validation logic:

```python
class LimitSearchWeb(Callback):
    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        if context.shared["search_web_count"] > self.max_calls:
            raise RuntimeError("Reached limit of `search_web` calls.")
```
!!! warning

    Raising an exception is the standard way to halt execution. This effectively acts as a 'circuit breaker' for your agent.

## Inspecting Data (`Context.current_span`)

The `Context.current_span` attribute provides access to the active trace span. This allows you to inspect (and modify) the data being processed, such as LLM inputs or Tool outputs.

Common attributes (available via `any_agent.tracing.attributes.GenAI`) include:

- `GenAI.INPUT_MESSAGES`: The chat history sent to the model.

- `GenAI.TOOL_NAME`: The name of the tool currently being executed.

- `GenAI.OUTPUT_MESSAGES`: The response received from the model.

## How it Works

When `agent.run()` or `agent.run_async()` executes, it triggers a series of events (e.g., before the LLM is called, after a tool is executed). You can register custom `Callback` classes to listen for these events.

### The Callback Contract

All callbacks share a strict contract: **They receive the current `Context` as input and must return a `Context` as output.**

```py
# pseudocode of an Agent run

history = [system_prompt, user_prompt]
context = Context()

for callback in agent.config.callbacks:
    # 1. Agent Start
    context = callback.before_agent_invocation(context)

while True:

    for callback in agent.config.callbacks:
        # 2. Pre-LLM
        context = callback.before_llm_call(context)

    response = CALL_LLM(history)

    for callback in agent.config.callbacks:
        # 3. Post-LLM
        context = callback.after_llm_call(context)

    history.append(response)

    if response.tool_executions:
        for tool_execution in tool_executions:
            # 4. Pre-Tool
            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context)

            tool_response = EXECUTE_TOOL(tool_execution)

            for callback in agent.config.callbacks:
                # 5. Post-Tool
                context = callback.after_tool_execution(context)

            history.append(tool_response)

    else:
        for callback in agent.config.callbacks:
            # 6. Agent DONE
            context = callback.after_agent_invocation(context)
        return response
```

Advanced designs such as safety guardrails or custom side-effects can be integrated into your agentic system using this functionality.

## Default Callbacks

`any-agent` comes with a set of default callbacks that will be used by default (if you don't pass a value to `AgentConfig.callbacks`):

- [`ConsolePrintSpan`][any_agent.callbacks.span_print.ConsolePrintSpan]

If you want to disable these default callbacks, you can pass an empty list:

```python
from any_agent import AgentConfig, AnyAgent
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="mistral:mistral-small-latest",
        instructions="Use the tools to find an answer",
        tools=[search_web, visit_webpage],
        callbacks=[]
    ),
)
```

## Registering your own Callbacks

Callbacks are provided to the agent using the [`AgentConfig.callbacks`][any_agent.config.AgentConfig.callbacks] property.

=== "Extending default callbacks"

    `any-agent` includes default callbacks (like console logging). Use [`get_default_callbacks`][any_agent.callbacks.get_default_callbacks] to keep them:

    ```py
    from any_agent import AgentConfig, AnyAgent
    from any_agent.callbacks import get_default_callbacks
    from any_agent.tools import search_web, visit_webpage

    agent = AnyAgent.create(
        "tinyagent",
        AgentConfig(
            model_id="gpt-4.1-nano",
            instructions="Use the tools to find an answer",
            tools=[search_web, visit_webpage],
            callbacks=[
                CountSearchWeb(),           # Custom callbacks first
                LimitSearchWeb(max_calls=3),
                *get_default_callbacks() #Runs after custom callbacks
            ]
        ),
    )
    ```

=== "Overriding default callbacks"

    To disable default logging or replace it entirely, pass a list without the defaults:

    ```py
    from any_agent import AgentConfig, AnyAgent
    from any_agent.tools import search_web, visit_webpage

    agent = AnyAgent.create(
        "tinyagent",
        AgentConfig(
            model_id="gpt-4.1-nano",
            instructions="Use the tools to find an answer",
            tools=[search_web, visit_webpage],
            callbacks=[
                CountSearchWeb(),
                LimitSearchWeb(max_calls=3)  # Default console logging disabled
            ]
        ),
    )
    ```

!!! warning

    Callbacks will be called in the order that they are added, so it is important to pay attention to the order in which you set the callback configuration.

    In the above example, passing:

    ```py
        callbacks=[
            LimitSearchWeb(max_calls=3) # ← This will fail!
            CountSearchWeb()    # Counter must come first
        ]
    ```

    Would fail because `context.shared["search_web_count"]` was not set yet.

## Examples

### Offloading sensitive information

Some inputs and/or outputs in your traces might contain sensitive information that you don't want
to be exposed in the [traces](../tracing.md).

You can use callbacks to offload the sensitive information to an external location and replace the span
attributes with a reference to that location:

```python
import json
from pathlib import Path

from any_agent.callbacks.base import Callback
from any_agent.callbacks.context import Context
from any_agent.tracing.attributes import GenAI

class SensitiveDataOffloader(Callback):

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:

        span = context.current_span

        if input_messages := span.attributes.get(GenAI.INPUT_MESSAGES):
            output_file = self.output_dir / f"{span.get_span_context().trace_id}.txt"
            output_file.write_text(str(input_messages))

            span.set_attribute(
                GenAI.INPUT_MESSAGES,
                json.dumps(
                    {"ref": str(output_file)}
                )
            )

        return context
```

You can find a working example in the [Callbacks Cookbook](../cookbook/callbacks.ipynb).

### Limit the number of steps

Some agent frameworks allow to limit how many steps an agent can take and some don't. In addition,
each framework defines a `step` differently: some count the llm calls, some the tool executions,
and some the sum of both.

You can use callbacks to limit how many steps an agent can take, and you can decide what to count
as a `step`:

```python
from any_agent.callbacks.base import Callback
from any_agent.callbacks.context import Context

class LimitLLMCalls(Callback):
    def __init__(self, max_llm_calls: int) -> None:
        self.max_llm_calls = max_llm_calls

    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:

        if "n_llm_calls" not in context.shared:
            context.shared["n_llm_calls"] = 0

        context.shared["n_llm_calls"] += 1

        if context.shared["n_llm_calls"] > self.max_llm_calls:
            raise RuntimeError("Reached limit of LLM Calls")

        return context

class LimitToolExecutions(Callback):
    def __init__(self, max_tool_executions: int) -> None:
        self.max_tool_executions = max_tool_executions

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:

        if "n_tool_executions" not in context.shared:
            context.shared["n_tool_executions"] = 0

        context.shared["n_tool_executions"] += 1

        if context.shared["n_tool_executions"] > self.max_tool_executions:
            raise RuntimeError("Reached limit of Tool Executions")

        return context
```
