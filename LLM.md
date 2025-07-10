# Any-Agent Library API Reference

## Overview
Any-Agent is a Python library providing a unified interface for building, running, evaluating, and serving AI agents across multiple frameworks. It supports 7+ agent frameworks with comprehensive tracing, tool integration, and serving capabilities.

## Installation
```bash
pip install any-agent
# With specific framework support:
pip install "any-agent[openai,langchain,llama_index]"
```

## Core API

### Main Classes

#### AnyAgent (Abstract Base)
```python
from any_agent import AnyAgent, AgentConfig, AgentFramework

# Create agent
agent = AnyAgent.create(
    agent_framework: AgentFramework | str,
    agent_config: AgentConfig
) -> AnyAgent

# Run agent
trace = agent.run(prompt: str, **kwargs) -> AgentTrace

# Async versions
agent = await AnyAgent.create_async(framework, config)
trace = await agent.run_async(prompt, **kwargs)

# Serve agent
agent.serve(serving_config=None)  # MCP or A2A
handle = await agent.serve_async(serving_config)
```

#### AgentConfig
```python
from any_agent import AgentConfig
from any_agent.tools import search_web, visit_webpage

config = AgentConfig(
    model_id: str,                    # Required: LiteLLM format
    api_base: str | None = None,
    api_key: str | None = None,
    name: str = "any_agent",
    description: str | None = None,
    instructions: str | None = None,  # System prompt
    tools: Sequence[Tool] = [],       # Tools list
    callbacks: list[Callback] = [],
    output_type: type[BaseModel] | None = None,  # Structured output
    # Framework-specific options:
    agent_type: Callable | None = None,
    agent_args: dict | None = None,
    model_type: Callable | None = None,
    model_args: dict | None = None    # temperature, etc.
)
```

#### AgentFramework (Enum)
```python
from any_agent import AgentFramework

# Supported frameworks:
AgentFramework.TINYAGENT    # Lightweight litellm-based
AgentFramework.OPENAI       # OpenAI agents library
AgentFramework.LANGCHAIN    # LangGraph-based
AgentFramework.LLAMA_INDEX  # Workflow-based
AgentFramework.GOOGLE       # Google ADK
AgentFramework.AGNO         # Agno framework
AgentFramework.SMOLAGENTS   # HuggingFace smolagents

# Convert from string
framework = AgentFramework.from_string("openai")
```

#### AgentTrace
```python
# Execution trace with spans and output
trace.final_output: str | dict | BaseModel | None
trace.spans: list[AgentSpan]
trace.tokens: TokenInfo      # Usage statistics
trace.cost: CostInfo         # Cost information

# Methods
trace.add_span(span: AgentSpan)
trace.spans_to_messages() -> list[AgentMessage]
```

## Built-in Tools

### Web Browsing
```python
from any_agent.tools import search_web, visit_webpage, search_tavily

# DuckDuckGo search
results = search_web(query: str) -> str

# Visit webpage and extract content
content = visit_webpage(url: str, timeout: int = 30) -> str

# Tavily search API
results = search_tavily(query: str, include_images: bool = False) -> str
```

### User Interaction
```python
from any_agent.tools import (
    show_plan, show_final_output, 
    ask_user_verification, send_console_message
)

show_plan(plan: str) -> str
show_final_output(answer: str) -> str
ask_user_verification(query: str) -> str
send_console_message(user: str, query: str) -> str
```

### Agent-to-Agent Communication
```python
from any_agent.tools import a2a_tool, a2a_tool_async

# Create A2A tool
tool = a2a_tool(
    url: str,                           # Agent URL
    toolname: str | None = None,        # Tool name
    http_kwargs: dict | None = None     # HTTP options
)

# Async version
tool = await a2a_tool_async(url, toolname, http_kwargs)
```

### Structured Output
```python
from any_agent.tools import FinalOutputTool
from pydantic import BaseModel

class MyOutput(BaseModel):
    answer: str
    confidence: float

# Create validation tool
output_tool = FinalOutputTool(output_type=MyOutput)
```

## Tool Configuration

### Tool Types
```python
from any_agent.config import Tool, MCPStdio, MCPSse

# Tool can be:
Tool = str | MCPParams | Callable[..., Any]

# Examples:
tools = [
    "search_web",              # Built-in tool name
    my_custom_function,        # Python function
    MCPStdio(...),            # MCP via stdio
    MCPSse(...)               # MCP via SSE
]
```

### MCP (Model Context Protocol)
```python
# MCP via stdio (Docker, local process)
mcp_stdio = MCPStdio(
    command: str,                              # Executable
    args: Sequence[str],                       # Arguments
    env: dict[str, str] | None = None,         # Environment
    tools: Sequence[str] | None = None,        # Tool filter
    client_session_timeout_seconds: float = 5
)

# MCP via Server-Sent Events
mcp_sse = MCPSse(
    url: str,                                  # Server URL
    headers: dict[str, str] | None = None,     # HTTP headers
    tools: Sequence[str] | None = None,        # Tool filter
    client_session_timeout_seconds: float = 5
)
```

## Complete Usage Example

```python
from any_agent import AnyAgent, AgentConfig, AgentFramework
from any_agent.tools import search_web, visit_webpage
from any_agent.callbacks import get_default_callbacks
from pydantic import BaseModel

# Define structured output
class ResearchOutput(BaseModel):
    summary: str
    sources: list[str]
    confidence: float

# Create agent configuration
config = AgentConfig(
    model_id="gpt-4o-mini",
    name="research_agent",
    instructions="You are a research assistant. Use web search to find accurate information.",
    tools=[search_web, visit_webpage],
    callbacks=get_default_callbacks(),
    output_type=ResearchOutput,
    model_args={"temperature": 0.1}
)

# Create agent
agent = AnyAgent.create(
    agent_framework=AgentFramework.OPENAI,
    agent_config=config
)

# Run agent
trace = agent.run("What are the latest developments in AI agents?")
print(f"Final output: {trace.final_output}")
print(f"Tokens used: {trace.tokens}")
print(f"Cost: ${trace.cost.total}")
```

## Environment Variables

```bash
# Model API keys (choose based on model_id)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export MISTRAL_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Tool API keys
export TAVILY_API_KEY="your-key"  # For search_tavily

# Tracing (optional)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

## Key Benefits

1. **Framework Agnostic**: Single API across 7+ frameworks
2. **Comprehensive Tooling**: Built-in web search, user interaction, A2A communication
3. **MCP Integration**: Model Context Protocol support for external tools
4. **Observability**: OpenTelemetry-based tracing with cost tracking
5. **Serving**: Deploy agents via MCP or A2A protocols
6. **Evaluation**: Built-in agent and LLM-based evaluation
7. **Structured Output**: Pydantic model validation
8. **Async Support**: Full async/await compatibility

This library provides everything needed to build, evaluate, and deploy production-ready AI agents across multiple frameworks with a single, consistent API.
