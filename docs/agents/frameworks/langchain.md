# LangChain

[https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

[https://github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)

## Default Agent Type

We use [`langgraph.prebuilt.create_react_agent`](https://langchain-ai.github.io/langgraph/reference/agents/?h=create_rea#langgraph.prebuilt.chat_agent_executor.create_react_agent) as default.
Check the reference to find additional supported `agent_args`.

## Default Model Type

We use [`any_llm`](https://mozilla-ai.github.io/any-llm/) as the default model provider.
Check the [AnyLLM documentation](https://mozilla-ai.github.io/any-llm/) for supported providers and `model_args`.

## Run args

Check [`RunnableConfig`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html) to find additional supported `AnyAgent.run` args.
