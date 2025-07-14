<p align="center">
  <picture>
    <img src="docs/images/any-agent-logo-mark.png" width="20%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

# any-agent

[![Docs](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/any-agent/actions/workflows/tests-integration.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/tests-integration.yaml/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A single interface to use and evaluate different agent frameworks.

</div>

## [Documentation](https://mozilla-ai.github.io/any-agent/)

- [Agents](https://mozilla-ai.github.io/any-agent/agents/)
- [Tools](https://mozilla-ai.github.io/any-agent/tools/)
- [Tracing](https://mozilla-ai.github.io/any-agent/tracing/)
- [Serving](https://mozilla-ai.github.io/any-agent/serving/)
- [Evaluation](https://mozilla-ai.github.io/any-agent/evaluation/)

## [Supported Frameworks](https://mozilla-ai.github.io/any-agent/)

[![Google ADK](https://img.shields.io/badge/Google%20ADK-4285F4?logo=google&logoColor=white)](https://github.com/google/adk-python) [![LangChain](https://img.shields.io/badge/LangChain-1e4545?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph) [![LlamaIndex](https://img.shields.io/badge/🦙%20LlamaIndex-fbcfe2)](https://github.com/run-llama/llama_index) [![OpenAI Agents](https://img.shields.io/badge/OpenAI%20Agents-black?logo=openai)](https://github.com/openai/openai-agents-python) [![Smolagents](https://img.shields.io/badge/Smolagents-ffcb3a?logo=huggingface&logoColor=white)](https://smolagents.org/) [![TinyAgents](https://img.shields.io/badge/TinyAgents-ffcb3a?logo=huggingface&logoColor=white)]([https://smolagents.org/](https://huggingface.co/blog/tiny-agents))  [Agno AI](https://docs.agno.com/introduction)


### Planned for Support (Contributions Welcome!)

[Open Github tickets for new frameworks](https://github.com/mozilla-ai/any-agent/issues?q=is%3Aissue%20state%3Aopen%20label%3Aframeworks)

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
For this example we use a model hosted by openai, but you may need to set the relevant API key for whichever provider being used.
See [our Model docs](https://mozilla-ai.github.io/any-agent/frameworks/#models) for more information about using different models.

```bash
export MISTRAL_API_KEY="YOUR_KEY_HERE"  # or OPENAI_API_KEY, etc
```

```python
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "tinyagent",  # See all options in https://mozilla-ai.github.io/any-agent/
    AgentConfig(
        model_id="mistral/mistral-small-latest",
        instructions="Use the tools to find an answer",
        tools=[search_web, visit_webpage]
    )
)

agent_trace = agent.run("Which Agent Framework is the best??")
print(agent_trace)
```


> [!TIP]
> Multi-agent can be implemented [using Agents-As-Tools](https://mozilla-ai.github.io/any-agent/agents/tools/#using-agents-as-tools).

## Cookbooks

Get started quickly with these practical examples:

- **[Creating your first agent](https://mozilla-ai.github.io/any-agent/cookbook/your_first_agent/)** - Build a simple agent with web search capabilities.
- **[Using Callbacks](https://mozilla-ai.github.io/any-agent/cookbook/callbacks/)** - Implement and use custom callbacks.
- **[Creating an agent with MCP](https://mozilla-ai.github.io/any-agent/cookbook/mcp_agent/)** - Integrate Model Context Protocol tools.
- **[Serve an Agent with A2A](https://mozilla-ai.github.io/any-agent/cookbook/serve_a2a/)** - Deploy agents with Agent-to-Agent communication.
- **[Building Multi-Agent Systems with A2A](https://mozilla-ai.github.io/any-agent/cookbook/a2a_as_tool/)** - Using an agent as a tool for another agent to interact with.

## Contributions

The AI agent space is moving fast! If you see a new agentic framework that AnyAgent doesn't yet support, we would love for you to create a Github issue. We also welcome your support in development of additional features or functionality.


## Running in Jupyter Notebook

If running in Jupyter Notebook you will need to add the following two lines before running AnyAgent, otherwise you may see the error `RuntimeError: This event loop is already running`. This is a known limitation of Jupyter Notebooks, see [Github Issue](https://github.com/jupyter/notebook/issues/3397#issuecomment-376803076)

```python
import nest_asyncio
nest_asyncio.apply()
```
