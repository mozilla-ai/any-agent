# Agent Instructions

`any-agent` provides 2 options to specify the instructions for your agent: `Import` and `String`.

In the first case, the import should point to a Python string.

!!! warning

    Some frameworks use complex default instructions for specific agent implementations.
    Completely replacing those instructions might result in unexpected behavior.

    In those cases, you might want to instead copy-paste and **extend** the default
    instructions.
    For example, check the [`CodeAgent` default instructions](https://github.com/huggingface/smolagents/blob/main/src/smolagents/prompts/code_agent.yaml) in `smolagents`.

=== "Import"

    For a variable that you would import like:

    ```python
    from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
    ```

    The expected syntax is `agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX`

    ```python
    from any_agent import AgentConfig, AgentFramework, AnyAgent

    framework = AgentFramework("openai")

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        instructions="agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX",
    )
    ```

=== "String"

    ```python
    from any_agent import AgentConfig, AgentFramework, AnyAgent

    framework = AgentFramework("openai")

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        instructions="You are a helpful assistant that can navigate the web",
        tools=[
            "any_agent.tools.search_web",
            "any_agent.tools.visit_webpage"
        ]
    )
    ```
