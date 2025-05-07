# Agent Tracing

`any-agent` uses [`openinference`](https://github.com/Arize-ai/openinference) to generate
standardized [OpenTelemetry](https://opentelemetry.io/) traces for any of the supported `Frameworks`.

An [`AgentTrace`][any_agent.tracing.trace.AgentTrace] is returned when calling [`agent.run`][any_agent.AnyAgent.run] or [`agent.run_async`][any_agent.AnyAgent.run_async].

## Example

By default, tracing to console and cost tracking is enabled. To configure tracing, pass a TracingConfig object [`TracingConfig`][any_agent.config.TracingConfig] when creating an agent.

```python
from any_agent import AgentConfig, AnyAgent, TracingConfig
from any_agent.tools import search_web

agent = AnyAgent.create(
        "openai",
        agent_config=AgentConfig(
                model_id="gpt-4o",
                tools=[search_web],
        ),
        tracing=TracingConfig(console=False)
      )
agent_trace = agent.run("Which agent framework is the best?")
```

### Console Output

Tracing will output standardized console output regardless of the
framework used.

```console
──────────────────────────────────────────────────────────────────────────── LLM ─────────────────────────────────────────────────────────────────────────────
input: Which agent framework is the best?
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────────── TOOL ────────────────────────────────────────────────────────────────────────────
tool_name: search_web
input: {'query': 'best agent framework 2023'}
╭────────────────────────────────────────────────────────────────────────── Output ──────────────────────────────────────────────────────────────────────────╮
│ Top 12 Open-Source Autonomous Agents & Agent Frameworks: The Future of ... The agent architecture came to life in March 2023, but it wasn't until a few    │
│ months later that it took a grip in the open-source community. The agent landscape may still seem like a "mad scientist" kind of experiment, but there are │
│ already a few insanely powerful models you can try. Top Open Source Autonomous Agents and Agent Frameworks Top 10 AI Agent Frameworks - gocodeo.com The    │
│ ultimate guide to AI agent frameworks, compare the best tools for building, scaling, and orchestrating intelligent systems. Features Pricing Docs Blog     │
│ Support. Install Now. Top 10 AI Agent Frameworks. Written By: April 4, 2025. We're well past the phase of "just prompt it and see what happens." As AI     │
│ agents inch closer to production ... List of Top 10 Multi-Agent Orchestrator Frameworks for Deploying AI ... 3. Bee Agent Framework (IBM) Introduction:    │
│ The Bee Agent Framework by IBM is a modular and enterprise-focused orchestration platform for managing large-scale multi-agent systems. It is designed to  │
│ integrate with IBM's AI solutions for optimized workflows and analytics. Features: Modular Architecture: Plug-and-play functionality for custom ... Top 9  │
│ AI Agent Frameworks as of April 2025 | Shakudo AutoGen is a framework developed by Microsoft that facilitates the creation of AI-powered applications by   │
│ automating the generation of code, models, and processes needed for complex workflows.It leverages large language models (LLMs) to help developers build,  │
│ fine-tune, and deploy AI solutions with minimal manual coding. AutoGen is particularly effective at automating the process of generating ... 10 best AI    │
│ agent frameworks - blog.apify.com Best AI agent framework platforms. AI agent frameworks are just one piece of the puzzle when it comes to building a      │
│ scalable, commercially viable AI application. Fully featured platforms do more than just offer tooling to facilitate agent development, they also make it  │
│ easier to integrate with third-party tools, handle cloud hosting, monitor ... Best 5 Frameworks To Build Multi-Agent AI Applications In this example, we   │
│ specify the prompt task as the code shows. Then, we create a new agent with reasoning=True to make it a thinking agent. When you run                       │
│ reasoning_ai_agent.py, you should see a result similar to the preview below.. 2. OpenAI Swarm. Swarm is an open-source, experimental agentic framework     │
│ recently released by OpenAI. It is a lightweight multi-agent orchestration framework. Agentic Framework Showdown: We Tested 8 AI Agent Frameworks They     │
│ reduce complexity and streamline decision-making as we build our agents. To find the best agentic framework for our client projects, we tested eight of    │
│ the most promising AI agent frameworks currently available, some relative newborns at less than six months from their first release: Autogen; CrewAI;      │
│ Langflow; LangGraph; LlamaIndex; n8n ... Comparing Open-Source AI Agent Frameworks - Langfuse Blog This post offers an in-depth look at some of the        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Spans

Here's what that returned trace spans would look like, accessible via the attribute `agent_trace.spans`:

```json
[
  {
    "name": "response",
    "context": {
      "trace_id": "0x1ee8d988d05d9c2e64a456dcccbf7a3c",
      "span_id": "0xd4a8cd952e71e1d1",
      "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0xbea970a46577575a",
    "start_time": "2025-04-07T10:20:25.327409Z",
    "end_time": "2025-04-07T10:20:26.813604Z",
    "status": {
      "status_code": "UNSET"
    },
    "attributes": {
      "llm.system": "openai",
      "output.mime_type": "application/json",
      "output.value": "{\"id\":\"resp_67f3a6e2d1dc8192b7d68b130f05f79801e7f8b6e38c7e7a\",\"created_at\":1744021218.0,\"error\":null,\"incomplete_details\":null,\"instructions\":\"Search the web to answer\",\"metadata\":{},\"model\":\"gpt-4o-2024-08-06\",\"object\":\"response\",\"output\":[{\"arguments\":\"{\\\"query\\\":\\\"best agent framework 2023\\\"}\",\"call_id\":\"call_xCZMfOtnbmKS1nGDywFtmCcR\",\"name\":\"search_web\",\"type\":\"function_call\",\"id\":\"fc_67f3a6e351988192a79ec42d68fccbe001e7f8b6e38c7e7a\",\"status\":\"completed\"}],\"parallel_tool_calls\":false,\"temperature\":1.0,\"tool_choice\":\"auto\",\"tools\":[{\"name\":\"search_web\",\"parameters\":{\"properties\":{\"query\":{\"description\":\"The search query to perform.\",\"title\":\"Query\",\"type\":\"string\"}},\"required\":[\"query\"],\"title\":\"search_web_args\",\"type\":\"object\",\"additionalProperties\":false},\"strict\":true,\"type\":\"function\",\"description\":\"Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results.\"}],\"top_p\":1.0,\"max_output_tokens\":null,\"previous_response_id\":null,\"reasoning\":{\"effort\":null,\"generate_summary\":null},\"status\":\"completed\",\"text\":{\"format\":{\"type\":\"text\"}},\"truncation\":\"disabled\",\"usage\":{\"input_tokens\":89,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":20,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":109},\"user\":null,\"store\":true}",
      "llm.tools.0.tool.json_schema": "{\"type\": \"function\", \"function\": {\"name\": \"search_web\", \"description\": \"Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results.\", \"parameters\": {\"properties\": {\"query\": {\"description\": \"The search query to perform.\", \"title\": \"Query\", \"type\": \"string\"}}, \"required\": [\"query\"], \"title\": \"search_web_args\", \"type\": \"object\", \"additionalProperties\": false}, \"strict\": true}}",
      "llm.token_count.completion": 89,
      "llm.token_count.prompt": 20,
      "llm.token_count.total": 109,
      "llm.output_messages.0.message.role": "assistant",
      "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_xCZMfOtnbmKS1nGDywFtmCcR",
      "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "search_web",
      "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{\"query\":\"best agent framework 2023\"}",
      "llm.input_messages.0.message.role": "system",
      "llm.input_messages.0.message.content": "Search the web to answer",
      "llm.model_name": "gpt-4o-2024-08-06",
      "llm.invocation_parameters": "{\"id\": \"resp_67f3a6e2d1dc8192b7d68b130f05f79801e7f8b6e38c7e7a\", \"created_at\": 1744021218.0, \"instructions\": \"Search the web to answer\", \"metadata\": {}, \"model\": \"gpt-4o-2024-08-06\", \"object\": \"response\", \"parallel_tool_calls\": false, \"temperature\": 1.0, \"tool_choice\": \"auto\", \"top_p\": 1.0, \"reasoning\": {}, \"status\": \"completed\", \"text\": {\"format\": {\"type\": \"text\"}}, \"truncation\": \"disabled\", \"store\": true}",
      "input.mime_type": "application/json",
      "input.value": "[{\"content\": \"Which agent framework is the best?\", \"role\": \"user\"}]",
      "llm.input_messages.1.message.role": "user",
      "llm.input_messages.1.message.content": "Which agent framework is the best?",
      "openinference.span.kind": "LLM"
    },
    "events": [],
    "links": [],
    "resource": {
      "attributes": {
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": "1.31.1",
        "service.name": "unknown_service"
      },
      "schema_url": ""
    }
  },
  {
    "name": "search_web",
    "context": {
      "trace_id": "0x1ee8d988d05d9c2e64a456dcccbf7a3c",
      "span_id": "0xe8fa92007caee376",
      "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0xbea970a46577575a",
    "start_time": "2025-04-07T10:20:26.821732Z",
    "end_time": "2025-04-07T10:20:28.420378Z",
    "status": {
      "status_code": "UNSET"
    },
    "attributes": {
      "llm.system": "openai",
      "tool.name": "search_web",
      "input.value": "{\"query\":\"best agent framework 2023\"}",
      "input.mime_type": "application/json",
      "output.value": "[Top 12 Open-Source Autonomous Agents & Agent Frameworks: The Future of ...](https://www.taskade.com/blog/top-autonomous-agents/)\nThe agent architecture came to life in March 2023, but it wasn't until a few months later that it took a grip in the open-source community. The agent landscape may still seem like a \"mad scientist\" kind of experiment, but there are already a few insanely powerful models you can try. Top Open Source Autonomous Agents and Agent Frameworks\n[Top 10 AI Agent Frameworks - gocodeo.com](https://www.gocodeo.com/post/top-10-ai-agent-frameworks)\nThe ultimate guide to AI agent frameworks, compare the best tools for building, scaling, and orchestrating intelligent systems. Features Pricing Docs Blog Support. Install Now. Top 10 AI Agent Frameworks. Written By: April 4, 2025. We're well past the phase of \"just prompt it and see what happens.\" As AI agents inch closer to production ...\n[List of Top 10 Multi-Agent Orchestrator Frameworks for Deploying AI ...](https://www.devopsschool.com/blog/list-of-top-10-multi-agent-orchestrator-frameworks-for-deploying-ai-agents/)\n3. Bee Agent Framework (IBM) Introduction: The Bee Agent Framework by IBM is a modular and enterprise-focused orchestration platform for managing large-scale multi-agent systems. It is designed to integrate with IBM's AI solutions for optimized workflows and analytics. Features: Modular Architecture: Plug-and-play functionality for custom ...\n[Top 9 AI Agent Frameworks as of April 2025 | Shakudo](https://www.shakudo.io/blog/top-9-ai-agent-frameworks)\nAutoGen is a framework developed by Microsoft that facilitates the creation of AI-powered applications by automating the generation of code, models, and processes needed for complex workflows.It leverages large language models (LLMs) to help developers build, fine-tune, and deploy AI solutions with minimal manual coding. AutoGen is particularly effective at automating the process of generating ...\n[10 best AI agent frameworks - blog.apify.com](https://blog.apify.com/10-best-ai-agent-frameworks/)\nBest AI agent framework platforms. AI agent frameworks are just one piece of the puzzle when it comes to building a scalable, commercially viable AI application. Fully featured platforms do more than just offer tooling to facilitate agent development, they also make it easier to integrate with third-party tools, handle cloud hosting, monitor ...\n[Best 5 Frameworks To Build Multi-Agent AI Applications](https://getstream.io/blog/multiagent-ai-frameworks/)\nIn this example, we specify the prompt task as the code shows. Then, we create a new agent with reasoning=True to make it a thinking agent. When you run reasoning_ai_agent.py, you should see a result similar to the preview below.. 2. OpenAI Swarm. Swarm is an open-source, experimental agentic framework recently released by OpenAI. It is a lightweight multi-agent orchestration framework.\n[Agentic Framework Showdown: We Tested 8 AI Agent Frameworks](https://www.willowtreeapps.com/craft/8-agentic-frameworks-tested)\nThey reduce complexity and streamline decision-making as we build our agents. To find the best agentic framework for our client projects, we tested eight of the most promising AI agent frameworks currently available, some relative newborns at less than six months from their first release: Autogen; CrewAI; Langflow; LangGraph; LlamaIndex; n8n ...\n[Comparing Open-Source AI Agent Frameworks - Langfuse Blog](https://langfuse.com/blog/2025-03-19-ai-agent-comparison)\nThis post offers an in-depth look at some of the leading open-source AI agent frameworks out there: LangGraph, the OpenAI Agents SDK, Smolagents, CrewAI, AutoGen, Semantic Kernel, and LlamaIndex agents. By the time you finish reading, you should have a clearer view of each framework's sweet spot, how they differ, and where they excel in real ...\n[Choosing the Right AI Agent Framework: LangGraph vs CrewAI vs OpenAI Swarm](https://www.relari.ai/blog/ai-agent-framework-comparison-langgraph-crewai-openai-swarm)\nWe chose LangGraph, CrewAI, and OpenAI Swarm because they represent the latest schools of thought in agent development. Here's a quick overview: LangGraph: As its name suggests, LangGraph bets on graph architecture as the best way to define and orchestrate agentic workflows. Unlike early versions of LangChain, LangGraph is a well designed framework with many robust and customizable features ...\n[Best AI Agent Frameworks](https://www.folio3.ai/blog/ai-agent-frameworks/)\nYour business demands the best AI agent framework to accelerate your project. It should support LLM integration, advanced reasoning, long-term memory, flexible tool coordination, and smooth collaboration between multiple agents. Here we discuss some AI agent frameworks that empower you to achieve unmatched levels of automation and intelligence.",
      "openinference.span.kind": "TOOL"
    },
    "events": [],
    "links": [],
    "resource": {
      "attributes": {
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": "1.31.1",
        "service.name": "unknown_service"
      },
      "schema_url": ""
    }
  },
```


### Dumping to File

The AgentTrace object is a pydantic model and can be saved to disk via standard pydantic practices:

```python
with open("output.json", "w", encoding="utf-8") as f:
  f.write(agent_trace.model_dump_json(indent=2))
```
