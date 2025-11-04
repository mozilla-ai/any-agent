# Model Configuration

## Overview

Model configuration in `any-agent` is designed to be consistent across all supported frameworks. We use [`any-llm`](https://mozilla-ai.github.io/any-llm/) as the default model provider, which acts as a unified interface allowing you to use any language model from any provider with the same syntax.

## Configuration Parameters

The model configuration is defined through several parameters in [`AgentConfig`][any_agent.config.AgentConfig]:

The `model_id` parameter selects which language model your agent will use. The format depends on the provider:

The `model_args` parameter allows you to pass additional arguments to the model, such as `temperature`, `top_k`, and other provider-specific parameters.

The `api_base` parameter allows you to specify a custom API endpoint. This is useful when:

- Using a local model server (e.g., Ollama, llama.cpp, llamafile)
- Routing through a proxy
- Using a self-hosted model endpoint

The `api_key` parameter allows you to explicitly specify an API key for authentication. By default, `any-llm` will automatically search for common environment variables (like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).å

See the [AnyLLM Provider Documentation](https://mozilla-ai.github.io/any-llm/providers/) for the complete list of supported providers.å
