# Agent Tracing

`any-agent` uses [`openinference`](https://github.com/Arize-ai/openinference) to generate
standardized [OpenTelemetry](https://opentelemetry.io/) traces for any of the supported [agent frameworks](./frameworks.md).

## Example

```py
from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tracing import get_trace_provider, setup_tracing

framework = AgentFramework("openai")
agent = AnyAgent(
        main_agent=AgentConfig(
        model_id="gpt-4o",
        tools=["any_agent.tools.search_web", "any_agent.tools.visit_webpage"]
    )
)

trace_provider = get_trace_provider(project_name="example", agent_framework=framework)
setup_tracing(trace_provider, framework)
```
