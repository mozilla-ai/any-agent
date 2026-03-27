---
title: Framework Comparison
description: Compare features across all agent frameworks supported by any-agent
head:
  - tag: style
    content: ".sl-markdown-content table { display: none; }"
  - tag: script
    attrs:
      src: /any-agent/scripts/framework-table.js
---

`any-agent` supports **{n}** agent frameworks. Click any framework to see its supported features and configuration.

<div class="framework-search-container">
  <input type="text" id="framework-search" class="framework-search-input" placeholder="Search frameworks..." />
</div>

<!-- FRAMEWORK-TABLE-START -->
| Framework | Docs | Callable Tools | MCP Tools | Composio Tools | Structured Output | Streaming | Multi-Agent (Handoffs) | Callbacks | any-llm Integration |
|-----------|------|----------------|-----------|----------------|-------------------|-----------|------------------------|-----------|---------------------|
| [Agno](/any-agent/agents/frameworks/agno/) | [docs](https://docs.agno.com/) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| [Google ADK](/any-agent/agents/frameworks/google-adk/) | [docs](https://google.github.io/adk-docs/) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| [LangChain](/any-agent/agents/frameworks/langchain/) | [docs](https://python.langchain.com/) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| [LlamaIndex](/any-agent/agents/frameworks/llama-index/) | [docs](https://docs.llamaindex.ai/) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| [OpenAI Agents SDK](/any-agent/agents/frameworks/openai/) | [docs](https://openai.github.io/openai-agents-python/) | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| [smolagents](/any-agent/agents/frameworks/smolagents/) | [docs](https://huggingface.co/docs/smolagents/) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| [TinyAgent](/any-agent/agents/frameworks/tinyagent/) | [docs](https://github.com/mozilla-ai/any-agent/blob/main/src/any_agent/frameworks/tinyagent.py) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
<!-- FRAMEWORK-TABLE-END -->
