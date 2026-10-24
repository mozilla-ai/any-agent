# Google Agent Development Kit (ADK)

[https://github.com/google/adk-python](https://github.com/google/adk-python)

## Default Agent Type

We use [`google.adk.agents.llm_agent.LlmAgent`](https://google.github.io/adk-docs/agents/llm-agents/) as default.
Check the reference to find additional supported `agent_args`.

## Default Model Type

We use [`any_llm`](https://docs.mozilla.ai/any-llm/) as the default model provider.
Check the [AnyLLM documentation](https://docs.mozilla.ai/any-llm/) for supported providers and `model_args`.

## Custom API endpoint

Set `AgentConfig.api_base` to select a custom endpoint. The Google ADK integration also accepts the legacy `model_args["api_base"]` setting; an explicit non-`None` `AgentConfig.api_base` takes precedence when both are provided. Model arguments are copied when preparing the model, so creating or reusing an agent does not modify the supplied mapping.

## Run args

Check [`RunConfig`](https://google.github.io/adk-docs/runtime/runconfig/) to find additional supported `AnyAgent.run` args.
