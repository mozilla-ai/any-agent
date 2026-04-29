<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/mozilla-ai/any-agent/refs/heads/main/docs/images/any-agent-logo-mark.png" width="20%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

# any-agent

[![Docs](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/any-agent/actions/workflows/tests-integration.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/tests-integration.yaml/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/any-agent)](https://pypi.org/project/any-agent/)
<a href="https://discord.gg/4gf3zXrQUc">
    <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
</a>

A single interface to use and evaluate different agent frameworks.

</div>

## [Documentation](https://docs.mozilla.ai/any-agent/)

- [Agents](https://docs.mozilla.ai/any-agent/agents/)
- [Tools](https://docs.mozilla.ai/any-agent/agents/tools/)
- [Tracing](https://docs.mozilla.ai/any-agent/core-concepts/tracing)
- [Serving](https://docs.mozilla.ai/any-agent/core-concepts/serving)
- [Evaluation](https://docs.mozilla.ai/any-agent/core-concepts/evaluation)

## [Supported Frameworks](https://docs.mozilla.ai/any-agent/)

[![TinyAgent](https://img.shields.io/badge/TinyAgent-ffcb3a?logo=huggingface&logoColor=white)](https://huggingface.co/blog/tiny-agents) [![Google ADK](https://img.shields.io/badge/Google%20ADK-4285F4?logo=google&logoColor=white)](https://github.com/google/adk-python) [![LangChain](https://img.shields.io/badge/LangChain-1e4545?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph) [![LlamaIndex](https://img.shields.io/badge/🦙%20LlamaIndex-fbcfe2)](https://github.com/run-llama/llama_index) [![OpenAI Agents](https://img.shields.io/badge/OpenAI%20Agents-black?logo=openai)](https://github.com/openai/openai-agents-python) [![Smolagents](https://img.shields.io/badge/Smolagents-ffcb3a?logo=huggingface&logoColor=white)](https://github.com/huggingface/smolagents) [![Agno AI](https://img.shields.io/badge/Agno-ff4017)](https://github.com/agno-agi/agno)



### Planned for Support (Contributions Welcome!)

[Open Github tickets for new frameworks](https://github.com/mozilla-ai/any-agent/issues?q=is%3Aissue%20state%3Aopen%20label%3Aframeworks)

> [!NOTE]
> The TinyAgent loop now lives in the standalone [`mozilla-ai-tinyagent`](https://github.com/mozilla-ai/tinyagent) package, which `any-agent` depends on. If you only need TinyAgent and don't need the multi-framework abstraction, install `mozilla-ai-tinyagent` directly for a leaner footprint.

## Requirements

- Python 3.11 or newer

## Quickstart

Refer to [pyproject.toml](./pyproject.toml) for a list of the options available.
Update your pip install command to include the frameworks that you plan on using:

```bash
pip install 'any-agent'
```

To define any agent system you will always use the same imports:

```python
from any_agent import AgentConfig, AnyAgent
```
For this example we use a model hosted by Mistral, but you may need to set the relevant API key for whichever provider being used.
See our [Model Configuration docs](https://docs.mozilla.ai/any-agent/agents/models/) for more information about configuring models.

```bash
export MISTRAL_API_KEY="YOUR_KEY_HERE"  # or OPENAI_API_KEY, etc
```

```python
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "tinyagent",  # See all options in https://docs.mozilla.ai/any-agent/
    AgentConfig(
        model_id="mistral:mistral-small-latest",
        instructions="Use the tools to find an answer",
        tools=[search_web, visit_webpage]
    )
)

agent_trace = agent.run("Which Agent Framework is the best??")
print(agent_trace)
```


> [!TIP]
> Multi-agent can be implemented [using Agents-As-Tools](https://docs.mozilla.ai/any-agent/agents/tools/#using-agents-as-tools).

## Cookbooks

Get started quickly with these practical examples:

- **[Creating your first agent](https://docs.mozilla.ai/any-agent/cookbook/your-first-agent)** - Build a simple agent with web search capabilities.
- **[Creating your first agent evaluation](https://docs.mozilla.ai/any-agent/cookbook/your-first-agent-evaluation)** - Evaluate that simple web search agent using 3 different methods.
- **[Using Callbacks](https://docs.mozilla.ai/any-agent/cookbook/callbacks)** - Implement and use custom callbacks.
- **[Creating an agent with MCP](https://docs.mozilla.ai/any-agent/cookbook/mcp-agent)** - Integrate Model Context Protocol tools.
- **[Serve an Agent with A2A](https://docs.mozilla.ai/any-agent/cookbook/serve-a2a)** - Deploy agents with Agent-to-Agent communication.
- **[Building Multi-Agent Systems with A2A](https://docs.mozilla.ai/any-agent/cookbook/a2a-as-tool)** - Using an agent as a tool for another agent to interact with.

## Contributions

The AI agent space is moving fast! If you see a new agentic framework that AnyAgent doesn't yet support, we would love for you to create a Github issue. We also welcome your support in development of additional features or functionality.


## Running in Jupyter Notebook

If running in Jupyter Notebook you will need to add the following two lines before running AnyAgent, otherwise you may see the error `RuntimeError: This event loop is already running`. This is a known limitation of Jupyter Notebooks, see [Github Issue](https://github.com/jupyter/notebook/issues/3397#issuecomment-376803076)

```python
import nest_asyncio
nest_asyncio.apply()
```
