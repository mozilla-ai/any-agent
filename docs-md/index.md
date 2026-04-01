# any-agent

:::caution
Compared to traditional code-defined workflows, agent frameworks introduce complexity,
additional security implications to consider, and demand much more computational power.

Before jumping to use one, carefully consider and evaluate how much value you
would get compared to manually defining a sequence of tools and LLM calls.
:::

## Requirements

- Python 3.11 or newer

## Installation

You can install the bare bones library as follows (only [TinyAgent](/any-agent/agents/frameworks/tinyagent/) will be available):

```bash
pip install any-agent
```

Or you can install it with the required dependencies for different frameworks:

```bash
pip install any-agent[agno,openai]
```

Refer to [pyproject.toml](https://github.com/mozilla-ai/any-agent/blob/main/pyproject.toml) for a list of the options available.

## Why any-agent

  - **Framework agnostic**: Switch between Agno, Google ADK, LangChain, LlamaIndex, OpenAI, smolagents, and TinyAgent with a single parameter change.

  - **Unified tracing**: Standardized OpenTelemetry traces across all frameworks for consistent observability.

  - **Built-in evaluation**: LLM-as-a-judge and agent-as-a-judge evaluation tools to assess agent performance.

  - **Serve anywhere**: Serve agents via A2A or MCP protocols and compose them as tools for other agents.

## For AI Systems

This documentation is available in two AI-friendly formats:

- **[llms.txt](https://mozilla-ai.github.io/any-agent/llms.txt)** - A structured overview with curated links to key documentation sections
- **[llms-full.txt](https://mozilla-ai.github.io/any-agent/llms-full.txt)** - Complete documentation content concatenated into a single file
