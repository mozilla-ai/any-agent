# any-agent

A single interface for different Agent frameworks


```py
from random import choice
from any_agent import AgentSchema, AgentFramework, load_agent, run_agent

agent = load_agent(
    framework=AgentFramework(choice(["langchain", "openai", "smolagents"]))
    main_agent=AgentSchema(model_id="gpt-4o-mini"),
)
result = run_agent(agent, "What day is today?")
```
