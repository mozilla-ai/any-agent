# Model Configuration

## Overview

Model configuration in `any-agent` is designed to be consistent across all supported frameworks. We use [any-llm](https://docs.mozilla.ai/any-llm/) as the default model provider, which acts as a unified interface allowing you to use any language model from any provider with the same syntax.

## Configuration Parameters

The model configuration is defined through several parameters in [AgentConfig](../api/config.md):

The `model_id` parameter selects which language model your agent will use. The format depends on the provider.

The `model_args` parameter allows you to pass additional arguments to the model, such as `temperature`, `top_k`, and other provider-specific parameters.

The `api_base` parameter allows you to specify a custom API endpoint. This is useful when:

- Using a local model server (e.g., Ollama, llmman, llama.cpp, llamafile)
- Routing through a proxy
- Using a self-hosted model endpoint

For example, [llmman](https://github.com/llmmanorg/llmman) serves the Ollama API on port `17434` instead of `11434`, so it can be used through the `ollama/` provider by pointing `api_base` at it:

```python
from any_agent import AgentConfig

config = AgentConfig(
    model_id="ollama/gemma4",
    api_base="http://localhost:17434",
)
```

The `api_key` parameter allows you to explicitly specify an API key for authentication. By default, `any-llm` will automatically search for common environment variables (like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

See the [AnyLLM Provider Documentation](https://docs.mozilla.ai/any-llm/providers/) for the complete list of supported providers.
