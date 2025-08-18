# any-agent

<p align="center">
  <picture>
    <img src="./images/any-agent-logo-mark.png" width="20%" alt="Project logo"/>
  </picture>
</p>

`any-agent` is a Python library providing a single interface to different agent frameworks.

!!! warning

    Compared to traditional code-defined workflows, agent frameworks introduce complexity,
    additional security implications to consider, and demand much more computational power.

    Before jumping to use one, carefully consider and evaluate how much value you
    would get compared to manually defining a sequence of tools and LLM calls.

## Requirements

- Python 3.11 or newer

## Installation

You can install the bare bones library as follows (only [`TinyAgent`](./agents/frameworks/tinyagent.md) will be available):

```bash
pip install any-agent
```

Or you can install it with the required dependencies for different frameworks:

```bash
pip install any-agent[agno,openai]
```

Refer to [pyproject.toml](https://github.com/mozilla-ai/any-agent/blob/main/pyproject.toml) for a list of the options available.

## Common Issues

### Sync vs Async Method Usage

When using `AnyAgent`, be careful about which method you use if you're in a sync vs async context.

```python
# ✅ Correct - sync method (no await needed)
agent = AnyAgent.create("tinyagent", config)
result = agent.run("What is the weather?")

# ✅ Correct - async method (await required)
agent = await AnyAgent.create_async("tinyagent", config)
result = await agent.run_async("What is the weather?")

# ❌ Incorrect - trying to await a sync method
result = await agent.run("What is the weather?")  # This will fail!

# ❌ Incorrect - not awaiting an async method
result = agent.run_async("What is the weather?")  # This returns a coroutine, not the result
```

### Jupyter Notebook Issues

If you're running in Jupyter Notebook, you may encounter `RuntimeError: This event loop is already running`. Add these lines before using any-agent:

```python
import nest_asyncio
nest_asyncio.apply()
```

## For AI Systems

This documentation is available in two AI-friendly formats:

- **[llms.txt](https://mozilla-ai.github.io/any-agent/llms.txt)** - A structured overview with curated links to key documentation sections
- **[llms-full.txt](https://mozilla-ai.github.io/any-agent/llms-full.txt)** - Complete documentation content concatenated into a single file
