from loguru import logger

try:
    from langchain.agents import AgentExecutor

    langchain_available = True
except ImportError:
    langchain_available = False


@logger.catch(reraise=True)
def run_langchain_agent(agent, query):
    if not langchain_available:
        raise ImportError("You need to `pip install langchain` to use this agent")
    executor = AgentExecutor(agent=agent, tools=agent.tools)
    executor.invoke(query)
