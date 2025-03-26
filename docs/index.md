# **Welcome to the any-agent docs**


```py
from random import choice
from any_agents import AgentSchema, load_agent, run_agent

agent = load_agent(
    framework=choice(["langchain", "openai", "smolagents"])
    main_agent=AgentSchema(model_id="gpt-4o-mini"),
)
result = run_agent(agent, "What day is today?")
```
