# Agent Evaluation

The any-agent evaluation module provides two primitives for evaluating agent traces using LLM-as-a-judge techniques:

- [`LlmJudge`][any_agent.evaluation.LlmJudge]: For evaluations that can be answered with a direct LLM call alongside the trace messages.
- [`AgentJudge`][any_agent.evaluation.AgentJudge]: For complex evaluations that utilize built-in and customizable tools to inspect specific parts of the trace and additional content

Both judges work with any-agent's unified tracing format and return structured evaluation results.

## Should you use an LlmJudge or AgentJudge?

Before automatically using an LLM based approach, it is worthwhile to consider whether it is necessary. For deterministic evaluations where you know exactly what to check, you may not want an LLM-based judge at all. Writing a custom evaluation function that directly examines the trace can be more efficient, reliable, and cost-effective. the any-agent [`AgentTrace`][any_agent.tracing.agent_trace.AgentTrace] provides a few helpful methods that can be used to extract common information.

### Example: Custom Evaluation Function

```python
from any_agent.tracing.agent_trace import AgentTrace

def evaluate_efficiency(trace: AgentTrace) -> dict:
    """Custom evaluation function for efficiency criteria."""

    # Direct access to trace properties
    token_count = trace.tokens.total_tokens
    step_count = len(trace.spans)
    final_output = trace.final_output

    # Apply your specific criteria
    results = {
        "token_efficient": token_count < 1000,
        "step_efficient": step_count <= 5,
        "has_output": final_output is not None,
        "token_count": token_count,
        "step_count": step_count
    }

    # Calculate overall pass/fail
    results["passed"] = all([
        results["token_efficient"],
        results["step_efficient"],
        results["has_output"]
    ])

    return results

# Usage
from any_agent import AgentConfig, AnyAgent
from any_agent.evaluation import LlmJudge
from any_agent.tools import search_web

# First, run an agent to get a trace
agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="gpt-4.1-nano",
        tools=[search_web]
    ),
)
trace = agent.run("What is the capital of France?")
evaluation = evaluate_efficiency(trace)
print(f"Evaluation results: {evaluation}")
```

### Working with Trace Messages

You can also examine the conversation flow directly:

```python
def check_tool_usage(trace: AgentTrace, required_tool: str) -> bool:
    """Check if a specific tool was used in the trace."""
    messages = trace.spans_to_messages()

    for message in messages:
        if message.role == "tool" and required_tool in message.content:
            return True
    return False

# Usage
used_search = check_tool_usage(trace, "search_web")
print(f"Used web search: {used_search}")
```

## LlmJudge

The `LlmJudge` is ideal for straightforward evaluation questions that can be answered by examining the complete trace text. It's efficient and works well for:

- Basic pass/fail assessments
- Simple criteria checking
- Text-based evaluations

### Example: Basic LLM Judge

```python
trace = agent.run("What is the capital of France?")

# Create and run the LLM judge
judge = LlmJudge(model_id="gpt-4o-mini")
result = judge.run(
    trace=trace,
    question="Did the agent provide the correct answer about the capital of France?"
)

print(f"Passed: {result.passed}")
print(f"Reasoning: {result.reasoning}")
```

## AgentJudge

The `AgentJudge` is designed for complex evaluations that require inspecting specific aspects of the trace. It comes equipped with evaluation tools and can accept additional custom tools for specialized assessments.

### Built-in Evaluation Tools

The `AgentJudge` automatically has access to these evaluation tools:

- `get_final_output()`: Get the agent's final output
- `get_tokens_used()`: Get total token usage
- `get_number_of_steps()`: Get number of steps taken
- `get_messages_from_trace()`: Get formatted trace messages

### Example: Agent Judge with Tool Access

```python
from any_agent.evaluation import AgentJudge

# Create an agent judge
judge = AgentJudge(model_id="gpt-4o")

# Evaluate with access to trace inspection tools
result = judge.run(
    trace=trace,
    question="Did the agent use web search and complete the task in under 5 steps?"
)

print(f"Passed: {result.passed}")
print(f"Reasoning: {result.reasoning}")
```

### Adding Custom Tools

You can extend the `AgentJudge` with additional tools for specialized evaluations:

```python
def check_for_ice_cream(message: str) -> bool:
    """Custom tool to check if a specific API was mentioned in a message

    Args:
        message: The message text to check for ice cream mentions

    Returns:
        True if ice cream is mentioned in the message, False otherwise
    """
    if 'ice cream' in message:
        return True
    else:
        return False

judge = AgentJudge(model_id="gpt-4o")
result = judge.run(
    trace=trace,
    question="Did the agent mention 'ice cream' in its final answer message?",
    additional_tools=[check_for_ice_cream]
)
```

## Custom Output Types

Both judges support custom output schemas using Pydantic models:

```python
from pydantic import BaseModel

class DetailedEvaluation(BaseModel):
    passed: bool
    reasoning: str
    confidence_score: float
    suggestions: list[str]

judge = LlmJudge(
    model_id="gpt-4o-mini",
    output_type=DetailedEvaluation
)

result = judge.run(trace=trace, question="Evaluate the agent's performance")
print(f"Confidence: {result.confidence_score}")
print(f"Suggestions: {result.suggestions}")
```
