# Using Callbacks

This cookbook shows you how to monitor, control, and secure your agents using callbacks.
We'll build three callbacks of increasing complexity: counting tool usage, enforcing
rate limits, and protecting sensitive data.

You can find more information about callbacks in the [docs](../agents/callbacks.md)

```python
%pip install 'any-agent' --quiet
%pip install ddgs --quiet

import warnings

import nest_asyncio

# Suppress technical warnings to reduce noise for the user
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

nest_asyncio.apply()
```

## Configure LLM Keys

For this tutorial, we'll use Mistral's mistral-small-latest (fast and affordable).
You could also use:
- `gpt-4o-mini`
- `claude-3-5-sonnet-latest`
- Any other model supported by any-agent

```python
import os
from getpass import getpass

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass("Enter your Mistral API Key: ")
```

```python
from any_agent import AgentConfig, AnyAgent
from any_agent.tools import search_web
```

## Running an agent (with default callbacks)

`any-agent` comes with a default callback that will always be used unless you pass a value to `AgentConfig.callbacks`:

- [`ConsolePrintSpan`](../api/callbacks.md)

```python
agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(model_id="mistral:mistral-small-latest", tools=[search_web]),
)

## Let's run a simple web search
agent_trace = agent.run("What are 5 LLM agent frameworks that are trending in 2025?")
```

## Count Tool Usage (`Context.shared`)

To control our agent, we first need to measure it. We will create a `StepCounter` callback.

**The Callback Contract:**
1. Callbacks receive a `context` object.
2. They can store data in `context.shared`.
3. They **must** return the `context` object.

```python
from any_agent.callbacks import Callback, Context
from any_agent.tracing.attributes import GenAI

class ToolUsageCounter(Callback):
    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        # 1. Initialize our counter if it doesn't exist yet
        if "tool_usage_count" not in context.shared:
            context.shared["tool_usage_count"] = 0

        # 2. Increment the counter
        context.shared["tool_usage_count"] += 1

        # 3. Print for visibility (optional)
        current_count = context.shared["tool_usage_count"]
        tool_name = context.current_span.attributes.get(GenAI.TOOL_NAME)
        print(f"🧮 Tracker: Tool '{tool_name}' called. (Count: {current_count})")

        # 4. MUST return context
        return context
```

## Enforce Rate Limits

Now that we are counting steps, we can act on that data.

We will create a `BudgetLimit` callback. If the tool usage exceeds our limit, we will raise an exception to immediately halt the agent. This prevents run-away costs.

```python
class BudgetLimit(Callback):
    def __init__(self, max_tools: int):
        self.max_tools = max_tools

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        # We can access the data set by the previous callback!
        current_count = context.shared.get("tool_usage_count", 0)

        if current_count > self.max_tools:
            msg = f"Exceeded limit of {self.max_tools} tool calls"
            raise RuntimeError(msg)

        return context
```

Let's put this all together.

1. We register our callbacks in `AgentConfig`.
2. We use `get_default_callbacks()` to keep the nice console logging.
3. We give the agent a **hard task** that requires multiple steps ("Find the weather in 3 different cities") to intentionally trigger our limit.

```python
from any_agent.callbacks import get_default_callbacks

# 1. Configure Agent with our custom stack
config = AgentConfig(
    model_id="mistral:mistral-small-latest",
    tools=[search_web],
    callbacks=[
        ToolUsageCounter(),  # Runs first: Counts the step
        BudgetLimit(
            max_tools=2
        ),  # Runs second: Checks the limit (set low to force a crash)
        *get_default_callbacks(),  # Runs last: Logs to console
    ],
)

agent = AnyAgent.create("tinyagent", config)

# 2. Run with a complex prompt
print("--- Starting Stress Test ---")
try:
    agent.run("Find the current weather in Tokyo, New York, and London.")
except RuntimeError as e:
    print(f"\n✅ Success! The agent was stopped: {e}")
```

## Bonus : Protect Sensitive Data

Beyond stopping the agent, callbacks can also **modify data** before it gets logged to your [traces](../tracing.md). This is critical for preventing Sensitive Information (PII) from leaking into your logs.

In the example below, we are going to implement a callback that:
1. Detects `INPUT_MESSAGES` in `Context.current_span`.
2. Writes this text to a secure local file.
3. **Replaces** the content in the current span with a reference link, so the trace remains clean.

```python
import json
from pathlib import Path

from any_agent.callbacks.base import Callback
from any_agent.callbacks.context import Context

class SensitiveDataOffloader(Callback):
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        span = context.current_span

        # 1. Check if we have input messages to scrub
        if input_messages := span.attributes.get(GenAI.INPUT_MESSAGES):
            # 2. Generate a secure filename based on the trace id
            output_file = self.output_dir / f"{span.get_span_context().trace_id}.txt"

            # 3. "Offload" the data to the secure location
            output_file.write_text(str(input_messages))

            # 4. Replace the span attribute with a reference
            span.set_attribute(
                GenAI.INPUT_MESSAGES, json.dumps({"ref": str(output_file)})
            )

        return context
```

We can now provide our callback to the agent.

You can find more information in [our docs](../agents/callbacks.md#providing-your-own-callbacks).

```python
from any_agent.callbacks import get_default_callbacks

agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="mistral:mistral-small-latest",
        tools=[search_web],
        callbacks=[SensitiveDataOffloader("sensitive-info"), *get_default_callbacks()],
    ),
)
agent_trace = agent.run("What are 5 LLM agent frameworks that are trending in 2025?")
```

```python
# Show that sensitive data was offloaded
import os

files = os.listdir("sensitive-info")
print(f"Created {len(files)} secure file(s)")

# Peek at what was saved (first 200 chars)
with open(f"sensitive-info/{files[0]}") as f:
    print(f"Offloaded data preview: {f.read()[:200]}...")
```

As you can see in the console output, the input messages in the trace have been now replaced by a reference to the external destination.
