---
title: Callbacks
description: Callback, Context, ConsolePrintSpan, and get_default_callbacks reference
---

## Callback

Base class for AnyAgent callbacks.

Subclass `Callback` and override any subset of the available lifecycle methods to observe, control, or extend agent execution without modifying the underlying agent implementation.

Each hook receives a `Context` object and should return that same `Context`, optionally after inspecting `context.current_span` or storing shared state in `context.shared`.

Common uses include logging, validation, guardrails, metrics collection, cost tracking, and intentional cancellation.

Base class for AnyAgent callbacks. Subclass and override any subset of the lifecycle methods.

### `Callback.before_agent_invocation()`

Will be called before the Agent invocation starts.

```python
def before_agent_invocation(
    self,
    context: Context,
    *args,
    **kwargs,
) -> Context
```

### `Callback.before_llm_call()`

Will be called before any LLM Call starts.

```python
def before_llm_call(
    self,
    context: Context,
    *args,
    **kwargs,
) -> Context
```

### `Callback.after_llm_call()`

Will be called after any LLM Call is completed.

```python
def after_llm_call(
    self,
    context: Context,
    *args,
    **kwargs,
) -> Context
```

### `Callback.before_tool_execution()`

Will be called before any Tool Execution starts.

```python
def before_tool_execution(
    self,
    context: Context,
    *args,
    **kwargs,
) -> Context
```

### `Callback.after_tool_execution()`

Will be called after any Tool Execution is completed.

```python
def after_tool_execution(
    self,
    context: Context,
    *args,
    **kwargs,
) -> Context
```

### `Callback.after_agent_invocation()`

Will be called once the Agent invocation ends.

```python
def after_agent_invocation(
    self,
    context: Context,
    *args,
    **kwargs,
) -> Context
```

---

## Context

Object that will be shared across callbacks.

Each AnyAgent.run has a separate `Context` available.

`shared` can be used to store and pass information across different callbacks.

Shared context object passed through all callbacks during an agent run.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `current_span` | `Span` | The active OpenTelemetry span with attributes (see GenAI) |
| `trace` | `AgentTrace` | Current execution trace |
| `tracer` | `Tracer` | OpenTelemetry tracer instance |
| `shared` | `dict[str, Any]` | Arbitrary shared state across callbacks |

---

## ConsolePrintSpan

Default callback that prints span information to the console using Rich formatting.

---

## `any_agent.callbacks.get_default_callbacks()`

Return instances of the default callbacks used in any-agent.

This function is called internally when the user doesn't provide a value for `AgentConfig.callbacks`.

```python
def get_default_callbacks() -> list[Callback]
```

**Returns:** A list of instances containing:
- `ConsolePrintSpan`
