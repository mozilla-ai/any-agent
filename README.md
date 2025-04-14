# any-agent

<div align="center">

[![Docs](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/any-agent/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/tests.yaml/)
[![Lint](https://github.com/mozilla-ai/any-agent/actions/workflows/lint.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/lint.yaml/)

[Documentation](https://mozilla-ai.github.io/any-agent/)

[Blog Post Introduction and Motivation](https://blog.mozilla.ai/introducing-any-agent-an-abstraction-layer-between-your-code-and-the-many-agentic-frameworks/)

</div>

`any-agent` is a Python library designed to provide a single interface to access many different agent frameworks.

Using `any-agent`, you can more easily switch to a new or different agent framework without needing to worry about the underlying API changes.

any-agent also provides a 'trace-first' [llm-as-a-judge powered evaluation tool](https://mozilla-ai.github.io/any-agent/evaluation/) for flexible evaluation of agent execution traces.

## Supported Frameworks

[Google ADK](https://github.com/google/adk-python) [![LangChain](https://img.shields.io/badge/LangChain-1e4545?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph) [![LlamaIndex](https://img.shields.io/badge/🦙%20LlamaIndex-fbcfe2)](https://github.com/run-llama/llama_index) [![OpenAI Agents](https://img.shields.io/badge/OpenAI%20Agents-black?logo=openai)](https://github.com/openai/openai-agents-python) [![Smolagents](https://img.shields.io/badge/Smolagents-ffcb3a?logo=huggingface&logoColor=white)](https://smolagents.org/) [Agno AI](https://docs.agno.com/introduction)

## Planned for Support (Contributions Welcome!)
[AWS Bedrock Agents](https://github.com/mozilla-ai/any-agent/issues/16),
[Pydantic AI](https://github.com/mozilla-ai/any-agent/issues/31),
[Microsoft AutoGen](https://github.com/mozilla-ai/any-agent/issues/30),
[Crew AI](https://github.com/mozilla-ai/any-agent/issues/17)


## Quickstart

Refer to pyproject.toml for a list of the extras available. Update your pip install command to include all of the frameworks that you plan on using.
```bash
pip install 'any-agent[smolagents,langchain,llama_index,openai,mcp]'
```

To define any agent system you will always use the same imports:

```py
from any_agent import AgentConfig, AgentFramework, AnyAgent

# Create the agent configuration for things like the underlying LLM as well as any tools.
main_agent = AgentConfig(
    model_id="gpt-4o-mini",
    tools=["any_agent.tools.search_web", "any_agent.tools.visit_webpage"]
)

# Choose one of the available frameworks:
from random import choice
framework = AgentFramework(
    choice(
        ["langchain", "llama_index", "openai", "smolagents"]
    )
)

# Create and run the agent:
agent = AnyAgent.create(framework, main_agent)
agent.run("Which Agent Framework is the best??")
```

`any-agent` supports the use of Model Context Protocol (MCP) servers, and if the agent framework allows,
any LLM and provider using [LiteLLM](https://docs.litellm.ai/docs/) syntax.

## Contributions

The AI agent space is moving fast! If you see a new agentic framework that AnyAgent doesn't yet support, we would love for you to create a Github issue. We also welcome your support in development of additional features or functionality.


## Running in Jupyter Notebook

If running in Jupyter Notebook you will need to add the following two lines before running AnyAgent, otherwise you may see the error `RuntimeError: This event loop is already running`. This is a known limitation of Jupyter Notebooks, see [Github Issue](https://github.com/jupyter/notebook/issues/3397#issuecomment-376803076)

```py
import nest_asyncio
nest_asyncio.apply()
```
