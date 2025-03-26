from loguru import logger

try:
    from agents import Runner, RunResult

    agents_available = True
except ImportError:
    agents_available = None


@logger.catch(reraise=True)
def run_openai_agent(agent, query) -> RunResult:
    if not agents_available:
        raise ImportError("You need to `pip install openai-agents` to use this agent")

    result = Runner.run_sync(agent, query)
    logger.info(result.final_output)
    return result
