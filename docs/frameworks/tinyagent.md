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
        model_id="gpt-4.1-nano",
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
