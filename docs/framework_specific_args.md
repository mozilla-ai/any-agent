# Framework Specific Arguments

The `agent_args` parameter in `any_agent.config.AgentConfig` allows you to pass arguments specific to the underlying framework that the agent instance is built on. 

**Example-1**: To pass the `output_type` parameter for structured output, when using the OpenAI Agents SDK:

```python
from pydantic import BaseModel
from any_agent import AgentConfig, AgentFramework, AnyAgent

class BookInfo(BaseModel):
    title: str
    author: str
    publication_year: int

framework = AgentFramework.OPENAI

agent = AnyAgent.create(
    framework,
    AgentConfig(
        model_id="gpt-4.1-mini",
        instructions="Extract book information from text",
        agent_args={
            "output_type": BookInfo
        }
    )
)

agent_trace = agent.run("The book is called 'The Alchemist' by Paulo Coelho and was published in 1988.")
print(agent_trace.final_output)
```

**Example-2**: In smolagents, for structured output one needs to use the `grammar` parameter. Additionally, `planning_interval` defines the interval at which the agent will run a planning step.

```python
from pydantic import BaseModel
from any_agent import AgentConfig, AgentFramework, AnyAgent


framework = AgentFramework.SMOLAGENTS

class WebPageInfo(BaseModel):
    title: str
    summary: str

agent = AnyAgent.create(
    framework,
    AgentConfig(
        model_id="gpt-4.1-mini",
        instructions="Extract webpage title and summary from url",
        agent_args={
            "planning_interval": 1,
            "grammar": WebPageInfo
        }
    )
)

agent_trace = agent.run("Could you get me the title and summary of the page at url 'https://blog.mozilla.ai/introducing-any-agent-an-abstraction-layer-between-your-code-and-the-many-agentic-frameworks/'?")
print(agent_trace.final_output)
```
