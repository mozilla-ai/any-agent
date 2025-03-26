from loguru import logger


@logger.catch(reraise=True)
def run_smolagents_agent(agent, query):
    result = agent.run(query)
    return result
