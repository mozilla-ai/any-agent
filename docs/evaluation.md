# Agent Evaluation

The any-agent evaluation module encourages three approaches for evaluating agent traces:

1. **Custom Code Evaluation**: Direct programmatic inspection of traces for deterministic checks
2. **[`LlmJudge`][any_agent.evaluation.LlmJudge]**: LLM-as-a-judge for evaluations that can be answered with a direct LLM call alongside a custom context
3. **[`AgentJudge`][any_agent.evaluation.AgentJudge]**: Complex LLM-based evaluations that utilize built-in and customizable tools to inspect specific parts of the trace or other custom information provided to the agent as a tool

## Choosing the Right Evaluation Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Custom Code** | Deterministic checks, performance metrics, specific criteria | Fast, reliable, cost-effective, precise control | Requires manual coding, limited to predefined checks |
| **LlmJudge** | Simple qualitative assessments, text-based evaluations | Easy to set up, flexible questions, good for subjective evaluation | Can be inconsistent, costs tokens, slower than code |
| **AgentJudge** | Complex multi-step evaluations, tool usage analysis | Most flexible, can support tool access to custom additional information sources | Highest cost, slowest, most complex setup |

Both judges work with any-agent's unified tracing format and return structured evaluation results.

## Custom Code Evaluation

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
        model_id="mistral/mistral-small-latest",
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

### Example: Evaluating Response Quality and Helpfulness

```python
from any_agent import AnyAgent, AgentConfig
from any_agent.tools import search_web
from any_agent.evaluation import LlmJudge
# Run an agent on a customer support task
agent = AnyAgent.create(
    "tinyagent",
    AgentConfig(
        model_id="mistral/mistral-small-latest",
        tools=[search_web]
    ),
)

trace = agent.run(
    "A customer is asking about setting up a new email account on the latest version of iOS. "
    "They mention they're not very tech-savvy and seem frustrated. "
    "Help them with clear, step-by-step instructions."
)

# Evaluate the quality of the agent's response multiple times
judge = LlmJudge(model_id="mistral/mistral-small-latest")
evaluation_questions = [
    "Did it provide clear, step-by-step instructions?",
    "Was the tone empathetic and appropriate for a frustrated, non-technical customer?",
    "Did it avoid using technical jargon without explanation?",
    "Was the response complete and actionable?",
    "Does the description specify which version of iOS this works with?"
]


# Run evaluation 4 times to check consistency
results = []
for evaluation_question in evaluation_questions:
    question = f"Evaluate whether the agent's response demonstrates good customer service by considering: {evaluation_question}."
    result = judge.run(context=str(trace.spans_to_messages()), question=evaluation_question)
    results.append(result)

# Print all results
for i, result in enumerate(results, 1):
    print(f"Run {i} - Passed: {result.passed}")
    print(f"Run {i} - Reasoning: {result.reasoning}")
    print("-" * 50)
```

!!! tip "Async Usage"
    For async applications, use `judge.run_async()` instead of `judge.run()`.

## AgentJudge

The `AgentJudge` is designed for complex evaluations that require inspecting specific aspects of the trace. It comes equipped with evaluation tools and can accept additional custom tools for specialized assessments.

### Built-in Evaluation Tools

The `AgentJudge` automatically has access to these evaluation tools:

- `get_final_output()`: Get the agent's final output
- `get_tokens_used()`: Get total token usage
- `get_steps_taken()`: Get number of steps taken
- `get_messages_from_trace()`: Get formatted trace messages
- `get_duration()`: Get the duration in seconds of the trace

### Example: Agent Judge with Tool Access

```python
from any_agent.evaluation import AgentJudge

# Create an agent judge
judge = AgentJudge(model_id="mistral/mistral-small-latest")

# Evaluate with access to trace inspection tools
eval_trace = judge.run(
    trace=trace,
    question="Does the final answer provided by the trace mention and correctly specify the most recent major version of iOS? You may need to do a web search to determine the most recent version of iOS. If the final answer does not mention the version at all, this criteria should fail",
    additional_tools=[search_web]
)

result = eval_trace.final_output
print(f"Passed: {result.passed}")
print(f"Reasoning: {result.reasoning}")
```

!!! tip "Async Usage"
    For async applications, use `judge.run_async()` instead of `judge.run()`.

### Adding Custom Tools

You can extend the `AgentJudge` with additional tools for specialized evaluations:

```python
def current_ios_version() -> str:
    """Custom tool to retrieve the most recent version of iOS

    Returns:
        The version of iOS
    """
    return "iOS 18.5"

judge = AgentJudge(model_id="mistral/mistral-small-latest")
eval_trace = judge.run(
    trace=trace,
    question="Does the final answer provided by the trace mention and correctly specify the most recent major version of iOS? If the final answer does not mention the version at all, this criteria should fail",
    additional_tools=[current_ios_version]
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
    model_id="mistral/mistral-small-latest",
    output_type=DetailedEvaluation
)

result = judge.run(trace=trace, question="Evaluate the agent's performance")
print(f"Confidence: {result.confidence_score}")
print(f"Suggestions: {result.suggestions}")
```
