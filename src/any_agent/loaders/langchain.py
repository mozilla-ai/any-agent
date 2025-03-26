from loguru import logger

from any_agent.schema import AgentSchema
from any_agent.tools.wrappers import import_and_wrap_tools, wrap_tool_langchain


try:
    from langchain.chat_models import init_chat_model
    from langchain.agents import create_tool_calling_agent

    langchain_available = True
except ImportError:
    langchain_available = False


@logger.catch(reraise=True)
def load_lanchain_agent(
    main_agent: AgentSchema, managed_agents: list[AgentSchema] | None = None
):
    if not langchain_available:
        raise ImportError(
            "You need to `pip install langchain langgraph` to use this agent"
        )

    if not main_agent.tools:
        main_agent.tools = [
            "any_agent.tools.search_web",
            "any_agent.tools.visit_webpage",
        ]

    if managed_agents:
        raise NotImplementedError("langchain managed agents are not supported yet")

    imported_tools = import_and_wrap_tools(main_agent.tools, wrap_tool_langchain)

    model = init_chat_model(main_agent.model_id)
    main_agent_instance = create_tool_calling_agent(
        model=model,
        tools=imported_tools,
    )
    return main_agent_instance
