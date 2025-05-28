# LangChain

[https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

[https://github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)

## Default Agent Type

We use [`langgraph.prebuilt.create_react_agent`](https://langchain-ai.github.io/langgraph/reference/agents/?h=create_rea#langgraph.prebuilt.chat_agent_executor.create_react_agent) as default.
Check the reference to find additional supported `agent_args`.

## Default Model Type

We use [`langchain_litellm.ChatLiteLLM`](https://python.langchain.com/docs/integrations/chat/litellm/#chatlitellm)
Check the reference to find additional supported `model_args`.

## Run args

Check [`RunnableConfig`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html) to find additional supported `AnyAgent.run` args.

## Examples

### Limiting the number of steps

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.tools import search_web, visit_webpage
from langchain_core.runnables import RunnableConfig

agent = AnyAgent.create(
    "langchain",
    AgentConfig(
        model_id="gpt-4.1-nano",
        instructions="You must use the available tools to find an answer",
        tools=[search_web, visit_webpage]
    )
)

agent.run(
    "Which Agent Framework is the best??",
    config=RunnableConfig(
        recursion_limit=3
    )
)
```
