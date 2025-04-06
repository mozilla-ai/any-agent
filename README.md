# any-agent

<div align="center">

[![Docs](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/any-agent/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/tests.yaml/)
[![Lint](https://github.com/mozilla-ai/any-agent/actions/workflows/lint.yaml/badge.svg)](https://github.com/mozilla-ai/any-agent/actions/workflows/lint.yaml/)

[Documentation](https://mozilla-ai.github.io/any-agent/)

</div>

`any-agent` is a Python library designed to provide a single interface to access many different agent frameworks.

Using `any-agent`, you can more easily switch to a new or different agent framework without needing to worry about the underlying API changes.

## Supported Frameworks

[![LangChain](https://img.shields.io/badge/LangChain-1e4545?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph) [![LlamaIndex](https://img.shields.io/badge/🦙%20LlamaIndex-fbcfe2)](https://github.com/run-llama/llama_index) [![OpenAI Agents](https://img.shields.io/badge/OpenAI%20Agents-black?logo=openai)](https://github.com/openai/openai-agents-python) [![Smolagents](https://img.shields.io/badge/Smolagents-ffcb3a?logo=huggingface&logoColor=white)](https://smolagents.org/)

[Coming Soon] [AWS Bedrock Agents](https://aws.amazon.com/bedrock/agents/)


## Quickstart

```py
from any_agent import AgentConfig, AgentFramework, AnyAgent
```

Create the agent configuration for things like the underlying LLM as well as any tools.

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

agent.run("Which Agent Framework is the best??")
```

`any-agent` supports the use of Model Context Protocol (MCP) servers, and if the agent framework allows,
any LLM and provider using [LiteLLM](https://docs.litellm.ai/docs/) syntax.

## Contributions

The AI agent space is moving fast! If you see a new agentic framework that AnyAgent doesn't yet support, we would love for you to create a Github issue. We also welcome your support in development of additional features or functionality.
