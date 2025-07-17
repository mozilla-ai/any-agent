# TinyAgent

As part of the bare bones library, we provide our own Python implementation based on [HuggingFace Tiny Agents](https://huggingface.co/blog/tiny-agents).

You can find it in [`any_agent.frameworks.tinyagent`](https://github.com/mozilla-ai/any-agent/blob/main/src/any_agent/frameworks/tinyagent.py).

## Examples

### Use MCP Tools

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.config import MCPStdio

agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="mistral/mistral-small-latest",
        instructions="You must use the available tools to find an answer",
        tools=[
            MCPStdio(
                command="uvx",
                args=["duckduckgo-mcp-server"]
            )
        ]
    )
)

result = agent.run(
    "Which Agent Framework is the best??"
)
print(result.final_output)
```

### Experimental any-llm Support

TinyAgent has experimental support for the [any-llm](https://github.com/mozilla-ai/any-llm) library, which provides a unified interface to different LLM providers. To enable this feature, set the `USE_ANY_LLM` environment variable:

```bash
export USE_ANY_LLM=1
```

You'll also need to install the any-llm dependency:

```bash
pip install 'any-agent[any_llm]'
```

When enabled, TinyAgent will use any-llm instead of LiteLLM for model completions

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.config import MCPStdio

# Set environment variable to enable any-llm
import os
os.environ["USE_ANY_LLM"] = "1"

agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="ollama/mistral-small3.2:latest",
        tools=[
            MCPStdio(
                command="uvx",
                args=["duckduckgo-mcp-server"]
            )
        ]
    )
)

result = agent.run(
    "Find me one new piece of AI news"
)
print(result.final_output)
```

See the [any-llm documentation](https://mozilla-ai.github.io/any-llm/providers) for a complete list of supported providers.
