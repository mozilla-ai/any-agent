# **any-agent**

`any-agent` is a Python library providing a single interface to different agent frameworks.

!!! warning

    Compared to traditional code-defined workflows, agent frameworks introduce complexity,
    additional security implications to consider, and demand much more computational power.

    Before jumping to use one, carefully consider and evaluate how much value you
    would get compared to manually defining a sequence of tools and LLM calls.

## Requirements

- Python 3.11 or newer

## Installation

You can install the bare bones library as follows (only [`TinyAgent`](./frameworks/tinyagent.md) will be available):

```bash
pip install any-agent
```

Or you can install it with the required dependencies for one of more frameworks:

```bash
pip install any-agent[agno,openai]
```

Refer to [pyproject.toml](https://github.com/mozilla-ai/any-agent/blob/main/pyproject.toml) for a list of the options available.
