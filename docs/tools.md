# Agent Tools

`any-agent` provides 3 options to specify what `tools` are available to your agent: `Import`, `Callable`, and `MCP` ([Model Context Protocol](https://modelcontextprotocol.io/introduction)).

You can use any combination of options in the same agent.

Under the hood, `any-agent` takes care of importing (for the first case) and wrapping (in any case needed) the
tool so it becomes usable by the selected framework.

=== "Import"

    For a tool that you would import like:

    ```py
    from any_agent.tools import search_web
    ```

    The expected syntax is `any_agent.tools.search_web`

    ```py
    from any_agent import AgentConfig, AgentFramework, AnyAgent

    framework = AgentFramework("openai")

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        tools=[
            "langchain_community.tools.TavilySearchResults",
            "any_agent.tools.visit_webpage"
        ]
    )
    ```

=== "Callable"

    ```py
    from any_agent import AgentConfig, AgentFramework, AnyAgent
    from langchain_community.tools import TavilySearchResults

    framework = AgentFramework("openai")

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        tools=[
            TavilySearchResults(
                max_results=3,
                include_raw_content=True,
            ),
        ]
    )
    ```

=== "MCP"

    ```py
    from any_agent import AgentConfig, AgentFramework, AnyAgent
    from any_agent.config import MCPTool

    framework = AgentFramework("openai")

    main_agent = AgentConfig(
        model_id="gpt-4o-mini",
        tools=[
            MCPTool(
                command="docker",
                args=["run", "-i", "--rm", "mcp/fetch"],
                tools=["fetch"]
            ),
        ]
    )
    ```
