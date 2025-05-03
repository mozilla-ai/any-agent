# Agent Instructions (aka System Prompt)

`any-agent` allows you to specify the instruction for the agent (often also referred to as a "system_prompt").

!!! warning

    Some frameworks use complex default instructions for specific agent implementations.
    Completely replacing those instructions might result in unexpected behavior.

    In those cases, you might want to instead copy-paste and **extend** the default
    instructions.
    For example, check the [`ToolCallingAgent` default instructions](https://github.com/huggingface/smolagents/blob/main/src/smolagents/prompts/toolcalling_agent.yaml) in `smolagents`.


```python
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from any_agent import AgentConfig

instruction = RECOMMENDED_PROMPT_PREFIX + "\nYou are a helpful assistant that can navigate the web."

main_agent = AgentConfig(
    model_id="gpt-4o-mini",
    instructions=instruction,
)
```
