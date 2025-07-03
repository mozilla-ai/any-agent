# Google ADK uses a different import path for LiteLLM, and smolagents uses the sync call
from any_agent.config import AgentFramework

LITELLM_IMPORT_PATHS = {
    AgentFramework.GOOGLE: "google.adk.models.lite_llm.acompletion",
    AgentFramework.LANGCHAIN: "litellm.acompletion",
    AgentFramework.TINYAGENT: "litellm.acompletion",
    AgentFramework.AGNO: "litellm.acompletion",
    AgentFramework.OPENAI: "litellm.acompletion",
    AgentFramework.SMOLAGENTS: "litellm.completion",
    AgentFramework.LLAMA_INDEX: "litellm.acompletion",
}
