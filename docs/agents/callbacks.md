# Agent Callbacks

For greater control when running your agent, `any-agent` includes support for custom [`Callbacks`][any_agent.callbacks.base.Callback] that
will be called at different points of the [`AnyAgent.run`][any_agent.AnyAgent.run]:

```py
# pseudocode of an Agent run

history = [system_prompt, user_prompt]
context = Context()

while True:

    for callback in agent.config.callbacks:
        context = callback.before_llm_call(context)

    response = CALL_LLM(history)

    for callback in agent.config.callbacks:
        context = callback.after_llm_call(context)

    history.append(response)

    if response.tool_executions:
        for tool_execution in tool_executions:

            for callback in agent.config.callbacks:
                context = callback.before_tool_execution(context)

            tool_response = EXECUTE_TOOL(tool_execution)

            for callback in agent.config.callbacks:
                context = callback.after_tool_execution(context)

            history.append(tool_response)

    else:
        return response
```

Advanced designs such as safety guardrails or custom side-effects can be integrated into your agentic system using this functionality.

## Context

During the agent run ( [`agent.run_async`][any_agent.AnyAgent.run_async] or [`agent.run`][any_agent.AnyAgent.run] ), a unique [`Context`][any_agent.callbacks.context.Context] object is created and shared across all callbacks.

`any-agent` populates the [`Context.current_span`][any_agent.callbacks.context.Context.current_span]
property so that callbacks can access information in a framework-agnostic way.

You can see what attributes are available for LLM Calls and Tool Executions by examining the [`GenAI`][any_agent.tracing.attributes.GenAI] class.

## Implementing Callbacks

All callbacks must inherit from the base [`Callback`][any_agent.callbacks.base.Callback] class and can choose to implement any subset of the available callback methods.

You can use [`Context.shared`][any_agent.callbacks.context.Context.shared] to store information meant
to be reused across callbacks:

```python
from any_agent.callbacks import Callback, Context
from any_agent.tracing.attributes import GenAI

class CountSearchWeb(Callback):
    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        if "search_web_count" not in context.shared:
            context.shared["search_web_count"] = 0
        if context.current_span.attributes[GenAI.TOOL_NAME] == "search_web":
            context.shared["search_web_count"] += 1
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

    Raising exceptions in callbacks will terminate the agent run immediately. Use this feature carefully to implement safety measures or validation logic.

## Default Callbacks

`any-agent` comes with a set of default callbacks that will be used by default (if you don't pass a value to `AgentConfig.callbacks`):

- [`AddCostInfo`][any_agent.callbacks.span_cost.AddCostInfo]
- [`ConsolePrintSpan`][any_agent.callbacks.span_print.ConsolePrintSpan]

If you want to disable these default callbacks, you can pass an empty list:

```python
from any_agent import AgentConfig, AnyAgent
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="mistral/mistral-small-latest",
        instructions="Use the tools to find an answer",
        tools=[search_web, visit_webpage],
        callbacks=[]
    ),
)
```

## Providing your own Callbacks

Callbacks are provided to the agent using the [`AgentConfig.callbacks`][any_agent.AgentConfig.callbacks] property.

=== "Alongside the default callbacks"

    You can use [`get_default_callbacks`][any_agent.callbacks.get_default_callbacks]:

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
                CountSearchWeb(),
                LimitSearchWeb(max_calls=3),
                *get_default_callbacks()
            ]
        ),
    )
    ```

=== "Override the default callbacks"

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
                LimitSearchWeb(max_calls=3)
            ]
        ),
    )
    ```

!!! warning

    Callbacks will be called in the order that they are added, so it is important to pay attention to the order
    in which you set the callback configuration.

    In the above example, passing:

    ```py
        callbacks=[
            LimitSearchWeb(max_calls=3)
            CountSearchWeb()
        ]
    ```

    Would fail because `context.shared["search_web_count"]`
    was not set yet.

## Example: Offloading sensitive information

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
