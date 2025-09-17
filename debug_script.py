import dotenv

from any_agent import AgentConfig, AnyAgent
from any_agent.tools import search_tavily

dotenv.load_dotenv()

agent = AnyAgent.create(
    "smolagents",
    AgentConfig(
        model_id="openai/gpt-4.1",
        instructions="You are a helpful assistant. Answer the question with available tools.",
        tools=[search_tavily],
        model_args={"tool_choice": "auto"},
    ),
)

trace = agent.run("Which agent framework is the best?")
