# TinyAgent

As part of the bare bones library, we provide our own Python implementation based on [HuggingFace Tiny Agents](https://huggingface.co/blog/tiny-agents). This implementation is based on our own [any-llm](https://github.com/mozilla-ai/any-llm) library which provides a unified interface to different LLM providers. See the [any-llm documentation](https://mozilla-ai.github.io/any-llm/providers) for a complete list of supported providers.

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
