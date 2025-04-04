# any-agent

`any-agent` is a Python library providing a single interface to different agent frameworks.

<div align="center">

[![Docs](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/any-agent/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/tests.yaml/)
[![Lint](https://github.com/mozilla-ai/any-agent/actions/workflows/lint.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/lint.yaml/)

[Documentation](https://mozilla-ai.github.io/any-agent/)

</div>

## Quickstart

```py
from any_agent import AgentConfig, AgentFramework, AnyAgent
```

Configure the agent:

```py
main_agent = AgentConfig(
    model_id="gpt-4o-mini",
    tools=["any_agent.tools.search_web", "any_agent.tools.visit_webpage"]
)
```

Chose one of the available frameworks:

```py
from random import choice

framework = AgentFramework(
    choice(
        ["langchain", "llama_index", "openai", "smolagents"]
    )
)
```

Create and run the agent:

```py
agent = AnyAgent.create(framework, main_agent)

agent.run("Which is the best Agent Framework?")
```
