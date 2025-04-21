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

## [Supported Frameworks](https://mozilla-ai.github.io/any-agent/frameworks/)

[![Google ADK](https://img.shields.io/badge/Google%20ADK-4285F4?logo=google&logoColor=white)](https://github.com/google/adk-python) [![LangChain](https://img.shields.io/badge/LangChain-1e4545?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph) [![LlamaIndex](https://img.shields.io/badge/🦙%20LlamaIndex-fbcfe2)](https://github.com/run-llama/llama_index) [![OpenAI Agents](https://img.shields.io/badge/OpenAI%20Agents-black?logo=openai)](https://github.com/openai/openai-agents-python) [![Smolagents](https://img.shields.io/badge/Smolagents-ffcb3a?logo=huggingface&logoColor=white)](https://smolagents.org/) [Agno AI](https://docs.agno.com/introduction)

### Planned for Support (Contributions Welcome!)
[AWS Bedrock Agents](https://github.com/mozilla-ai/any-agent/issues/16),
[Pydantic AI](https://github.com/mozilla-ai/any-agent/issues/31),
[Microsoft AutoGen](https://github.com/mozilla-ai/any-agent/issues/30),
[Crew AI](https://github.com/mozilla-ai/any-agent/issues/17)

## Quickstart

Refer to [pyproject.toml](./pyproject.toml) for a list of the options available.
Update your pip install command to include the frameworks that you plan on using (or use `all` to install all the currently supported):

```bash
pip install 'any-agent[all]'
```

To define any agent system you will always use the same imports:

```py
from any_agent import AgentConfig, AgentFramework, AnyAgent
from any_agent.tracing import setup_tracing  # Optional, but recommended

# See all options in https://mozilla-ai.github.io/any-agent/frameworks/
framework = "smolagents"

setup_tracing(framework)
```

### Single agent

```py
from any_agent.tools import search_web, visit_webpage
agent = AnyAgent.create(
    framework,
    AgentConfig(
        model_id="gpt-4.1-nano",
        instructions="Use the tools to find an answer",
        tools=[search_web, visit_webpage]
    )
)

agent.run("Which Agent Framework is the best??")
```

### Multi-agent

```py
from any_agent.tools import search_web, visit_webpage
agent = AnyAgent.create(
    framework,
    AgentConfig(
        model_id="gpt-4.1-mini",
        instructions="You are the main agent. Use the other available agents to find an answer",
    ),
    managed_agents=[
        AgentConfig(
            name="search_web_agent",
            description="An agent that can search the web",
            model_id="gpt-4.1-nano",
            tools=[search_web]
        ),
        AgentConfig(
            name="visit_webpage_agent",
            description="An agent that can visit webpages",
            model_id="gpt-4.1-nano",
            tools=[visit_webpage]
        )
    ]
)

agent.run("Which Agent Framework is the best??")
```

## Features

`any-agent` supports the use of Model Context Protocol (MCP) servers, and if the agent framework allows,
any LLM and provider using [LiteLLM](https://docs.litellm.ai/docs/) syntax.

Learn more in the docs:

- [Models](https://mozilla-ai.github.io/any-agent/frameworks/#models)
- [Tools](https://mozilla-ai.github.io/any-agent/tools/)
- [Instructions](https://mozilla-ai.github.io/any-agent/instructions/)
- [Tracing](https://mozilla-ai.github.io/any-agent/tracing/)
- [Evaluation](https://mozilla-ai.github.io/any-agent/evaluation/)


## Contributions

The AI agent space is moving fast! If you see a new agentic framework that AnyAgent doesn't yet support, we would love for you to create a Github issue. We also welcome your support in development of additional features or functionality.


## Running in Jupyter Notebook

If running in Jupyter Notebook you will need to add the following two lines before running AnyAgent, otherwise you may see the error `RuntimeError: This event loop is already running`. This is a known limitation of Jupyter Notebooks, see [Github Issue](https://github.com/jupyter/notebook/issues/3397#issuecomment-376803076)

```py
import nest_asyncio
nest_asyncio.apply()
```
