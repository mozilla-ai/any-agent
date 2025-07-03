# Agent Callbacks

`any-agent` allow you to provide custom [`Callbacks`][any_agent.callbacks.base.Callback] that
will be called at different points of the [`AnyAgent.run`][any_agent.AnyAgent.run]:

- [`before_llm_call`][any_agent.callbacks.base.Callback.before_llm_call]
- [`after_llm_call`][any_agent.callbacks.base.Callback.after_llm_call]
- [`before_tool_execution`][any_agent.callbacks.base.Callback.before_tool_execution]
- [`after_tool_execution`][any_agent.callbacks.base.Callback.after_tool_execution]

Each separate agent run share an unique [`Context`][any_agent.callbacks.context.Context] object
across all callbacks.

`any-agent` takes care of populating the [`Context.current_span`][any_agent.callbacks.context.Context.current_span]
property so the callbacks can access information in a framework-agnostic way. You can check the attributes available
for LLM Calls and Tool Executions in the [example spans](../tracing.md#spans).

## Implementing Callbacks

All callbacks must inherit from the base [`Callback`][any_agent.callbacks.base.Callback] class and
 only need to implement the methods they need:

```python
from any_agent.callbacks.base import Callback
from any_agent.callbacks.context import Context

class CountSearchWeb(Callback):
    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        current_span = context.current_span
        if "search_web_count" not in context.shared:
            context.shared["search_web_count"] = 0
        if current_span.attributes["gen_ai.tool.name"] == "search_web":
            context.shared["search_web_count"] += 1

class LimitSearchWeb(Callback):
    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        if context.shared["search_web_count"] > self.max_calls:
            raise RuntimeError("Reached limit of `search_web` calls.")
```

## Providing Callbacks

You can provide callbacks to the agent using the [`AgentConfig.callbacks`] property:

```python
from any_agent import AgentConfig

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

    The order of the callbacks matter.

    In the above example, passing:

    ```py
        callbacks=[
            LimitSearchWeb(max_calls=3)
            CountSearchWeb()
        ]
    ```

    Would fail during the first call because `context.shared["search_web_count"]`
    was not set yet.
