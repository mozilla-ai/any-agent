from loguru import logger

from any_agent.schema import AgentFramework, AgentSchema
from any_agent.tools.wrappers import import_and_wrap_tools


try:
    from langchain.chat_models import init_chat_model
    from langgraph.prebuilt import create_react_agent

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

    imported_tools, _ = import_and_wrap_tools(
        main_agent.tools, agent_framework=AgentFramework.LANGCHAIN
    )

    model = init_chat_model(main_agent.model_id)

    main_agent_instance = create_react_agent(
        model=model, tools=imported_tools, prompt=main_agent.instructions
    )

    return main_agent_instance
