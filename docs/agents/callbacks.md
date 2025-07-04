# Agent Callbacks

For greater control when running your agent, `any-agent` includes support for custom [`Callbacks`][any_agent.callbacks.base.Callback] that
will be called at different points of the [`AnyAgent.run`][any_agent.AnyAgent.run]:

```py
# pseudocode of an Agent run

history = [system_prompt, user_prompt]
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

Each time an agent is run ( [`agent.run_async`][any_agent.AnyAgent.run_async] or [`agent.run`][any_agent.AnyAgent.run] ), a unique [`Context`][any_agent.callbacks.context.Context] object
is shared across all callbacks.

All any-agent agents will populate the [`Context.current_span`][any_agent.callbacks.context.Context.current_span]
property so that the callbacks can access information in a framework-agnostic way. You can check the attributes available
for LLM Calls and Tool Executions in the [example spans](../tracing.md#spans).

## Implementing Callbacks

All callbacks must inherit from the base [`Callback`][any_agent.callbacks.base.Callback] class and
 can choose to implement any subset of the available callback methods.:

```python
from any_agent.callbacks import Callback, Context

class CountSearchWeb(Callback):
    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        if "search_web_count" not in context.shared:
            context.shared["search_web_count"] = 0
        if context.current_span.attributes["gen_ai.tool.name"] == "search_web":
            context.shared["search_web_count"] += 1

class LimitSearchWeb(Callback):
    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        if context.shared["search_web_count"] > self.max_calls:
            raise RuntimeError("Reached limit of `search_web` calls.")
```

## Providing Callbacks

These callbacks are provided to the agent using the [`AgentConfig.callbacks`][any_agent.AgentConfig.callbacks] property:

```python
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

## Error Handling in Callbacks

Callbacks can raise exceptions to stop agent execution. This is useful for implementing safety guardrails or validation logic:

```python
class SafetyGuard(Callback):
    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        tool_name = context.current_span.attributes.get("gen_ai.tool.name", "")

        # Block dangerous tools
        if tool_name in ["delete_file", "execute_code"]:
            raise RuntimeError(f"Tool '{tool_name}' is not allowed for safety reasons")

        return context

class ContentFilter(Callback):
    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        output = context.current_span.attributes.get("gen_ai.output", "")

        # Check for inappropriate content
        inappropriate_words = ["spam", "malware", "hack"]
        if any(word in output.lower() for word in inappropriate_words):
            raise RuntimeError("Generated content contains inappropriate language")

        return context
```

!!! warning

    Raising exceptions in callbacks will terminate the agent run immediately. Use this feature carefully to implement safety measures or validation logic.
