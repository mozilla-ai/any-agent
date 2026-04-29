# TinyAgent

A small Python implementation of an agent loop based on [HuggingFace Tiny Agents](https://huggingface.co/blog/tiny-agents).

The loop lives in the standalone [`mozilla-ai-tinyagent`](https://github.com/mozilla-ai/tinyagent) package. `any-agent` re-exposes it through the `AgentFramework.TINYAGENT` backend so that switching between TinyAgent and any other framework remains a single-parameter change.

If you only need the TinyAgent loop and don't need the multi-framework abstraction, install [`mozilla-ai-tinyagent`](https://pypi.org/project/mozilla-ai-tinyagent/) directly for a leaner footprint.

## Examples

### Use MCP Tools

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.config import MCPStdio

agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="mistral:mistral-small-latest",
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
