# LlamaIndex

[https://github.com/run-llama/llama_index](https://github.com/run-llama/llama_index)

## Default Agent Type

We use [`llama_index.core.agent.workflow.react_agent.FunctionAgent`](https://docs.llamaindex.ai/en/stable/api_reference/agent/#llama_index.core.agent.workflow.FunctionAgent) as default.
However, this agent requires that tools are used. If no tools are used, any-agent will default to [`llama_index.core.agent.workflow.react_agent.ReActAgent`](https://docs.llamaindex.ai/en/stable/api_reference/agent/#llama_index.core.agent.workflow.ReActAgent).
Check the reference to find additional supported `agent_args`.

## Default Model Type

We use [`llama_index.llms.litellm.LiteLLM`](https://docs.llamaindex.ai/en/stable/examples/llm/litellm/) as default.
Check the reference to find additional supported `model_args`.

## Examples

### Limiting the number of steps

Pending on https://github.com/run-llama/llama_index/issues/18535
