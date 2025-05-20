# Agent Tracing

`any-agent` generates
standardized [OpenTelemetry](https://opentelemetry.io/) traces for any of the supported `Frameworks`,
based on the [Semantic conventions for generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

An [`AgentTrace`][any_agent.tracing.trace.AgentTrace] is returned when calling [`agent.run`][any_agent.AnyAgent.run] or [`agent.run_async`][any_agent.AnyAgent.run_async].

## Example

By default, tracing to console and cost tracking is enabled. To configure tracing, pass a TracingConfig object [`TracingConfig`][any_agent.config.TracingConfig] when creating an agent.

```python
from any_agent import AgentConfig, AnyAgent, TracingConfig
from any_agent.tools import search_web

agent = AnyAgent.create(
        "openai",
        agent_config=AgentConfig(
                model_id="gpt-4o",
                tools=[search_web],
        ),
        tracing=TracingConfig(console=False)
      )
agent_trace = agent.run("Which agent framework is the best?")
```

### Console Output

Tracing will output standardized console output regardless of the
framework used.

```console
```

### Spans

Here's what that returned trace spans would look like, accessible via the attribute `agent_trace.spans`:

```json
```


### Dumping to File

The AgentTrace object is a pydantic model and can be saved to disk via standard pydantic practices:

```python
with open("output.json", "w", encoding="utf-8") as f:
  f.write(agent_trace.model_dump_json(indent=2))
```
