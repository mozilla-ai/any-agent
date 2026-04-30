# One interface. Every agent framework.

{% hint style="warning" %}
**`any-agent` is in soft deprecation.** It started as a research project to compare agent frameworks and refine a minimal common surface. That distillation has graduated to its own package: [`mozilla-ai-tinyagent`](https://github.com/mozilla-ai/tinyagent) (PyPI: `mozilla-ai-tinyagent`). For new projects we recommend using `tinyagent` directly.

`any-agent` continues to be published and we'll take security/bug-fix PRs, but no new features are planned. Reach for `any-agent` when you specifically need to **run or evaluate agents across multiple frameworks** (Agno, Google ADK, LangChain, LlamaIndex, OpenAI Agents SDK, smolagents) under one API. If you only need the core agent loop, [`mozilla-ai-tinyagent`](https://pypi.org/project/mozilla-ai-tinyagent/) is the leaner path.
{% endhint %}

## One interface. Every agent framework.

any-agent gives you a single interface for building agents across multiple frameworks.

Choose a path:

{% content-ref url="cookbook/your-first-agent.md" %}
[Your First Agent](cookbook/your-first-agent.md)
{% endcontent-ref %}

{% content-ref url="agents/index.md" %}
[Define and Run Agents](agents/index.md)
{% endcontent-ref %}

[View on GitHub](https://github.com/mozilla-ai/any-agent)

## Why any-agent

- **Framework agnostic**: Switch between Agno, Google ADK, LangChain, LlamaIndex, OpenAI, smolagents, and TinyAgent with a single parameter change.
- **Unified tracing**: Standardized OpenTelemetry traces across all frameworks for consistent observability.
- **Built-in evaluation**: LLM-as-a-judge and agent-as-a-judge evaluation tools to assess agent performance.
- **Serve anywhere**: Serve agents via A2A or MCP protocols and compose them as tools for other agents.

## Requirements

- Python 3.11 or newer

## Installation

You can install the bare bones library as follows (only [TinyAgent](agents/frameworks/tinyagent.md) will be available):

```bash
pip install any-agent
```

Or you can install it with the required dependencies for different frameworks:

```bash
pip install any-agent[agno,openai]
```

Refer to [pyproject.toml](https://github.com/mozilla-ai/any-agent/blob/main/pyproject.toml) for a list of the options available.

## For AI Systems

This documentation is available in two AI-friendly formats:

- **[llms.txt](https://docs.mozilla.ai/any-agent/llms.txt)** - A structured overview with curated links to key documentation sections
- **[llms-full.txt](https://docs.mozilla.ai/any-agent/llms-full.txt)** - Complete documentation content concatenated into a single file
