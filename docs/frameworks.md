# Agent Frameworks

Here you can find the frameworks currently supported in `any-agent`, along with some basic examples.

!!! info

    If you are missing any agent framework, check the [existing issues](https://github.com/mozilla-ai/any-agent/issues?q=is%3Aissue%20state%3Aopen%20label%3Aframeworks)
    to see if it has been already requested and comment/upvote on that issue.

    If there is no existing issue, don't hesitate to request and/or contribute it.

=== "🦜🔗 LangChain"

    [LangChain Repo](https://github.com/langchain-ai/langchain)

    ``` py
    agent = AnyAgent.create(
        framework=AgentFramework("langchain"),
        main_agent=AgentConfig(
            model_id="gpt-4o-mini"
        )
    )
    agent.run("Which is the best Agent Framework?")
    ```

=== "🗂️ LlamaIndex 🦙"

    [LLamaIndex Repo](https://github.com/run-llama/llama_index)

    ``` py
    agent = AnyAgent.create(
        framework=AgentFramework("llama_index"),
        main_agent=AgentConfig(
            model_id="gpt-4o-mini"
        )
    )
    agent.run("Which is the best Agent Framework?")
    ```

=== "OpenAI Agents"

    [OpenAI Agents Repo](https://github.com/openai/openai-agents-python)

    ``` py
    agent = AnyAgent.create(
        framework=AgentFramework("openai"),
        main_agent=AgentConfig(
            model_id="gpt-4o-mini"
        )
    )
    agent.run("Which is the best Agent Framework?")
    ```

=== "🤗 smolagents"

    [smolagents Repo](https://github.com/huggingface/smolagents)

    ``` py
    agent = AnyAgent.create(
        framework=AgentFramework("smolagents"),
        main_agent=AgentConfig(
            model_id="gpt-4o-mini"
        )
    )
    agent.run("Which is the best Agent Framework?")
    ```
