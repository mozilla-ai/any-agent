# Evaluating your first agent

In this tutorial, we'll build upon the web search agent from [your_first_agent.ipynb](https://github.com/mozilla-ai/any-agent/blob/main/docs/cookbook/your_first_agent.ipynb) and demonstrate how to evaluate its performance using any-agent's evaluation framework. We'll explore different evaluation methods including custom code evaluation, an LLM-based judge, and an agent-based judge.

Note: Since we are building on the previous notebook, we encourage you to run that one first to read through details and choices available while building the agent before evaluating it.

## Install Dependencies

any-agent uses the python asyncio module to support async functionality. When running in Jupyter notebooks, this means we need to enable the use of nested event loops. We'll install any-agent and enable this below using nest_asyncio.

```python
%pip install 'any-agent' --quiet
%pip install ddgs --quiet

import warnings

import nest_asyncio

# Suppress technical warnings to reduce noise for the user
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

nest_asyncio.apply()
```

## Set Up the Web Search Agent

First, let's recreate the web search agent from [the previous tutorial](https://github.com/mozilla-ai/any-agent/blob/main/docs/cookbook/your_first_agent.ipynb) so we have something to evaluate.

```python
import os
from getpass import getpass

if "MISTRAL_API_KEY" not in os.environ:
    print("MISTRAL_API_KEY not found in environment!")
    api_key = getpass("Please enter your MISTRAL_API_KEY: ")
    os.environ["MISTRAL_API_KEY"] = api_key
    print("MISTRAL_API_KEY set for this session!")
else:
    print("MISTRAL_API_KEY found in environment.")
```

```python
from any_agent import AgentConfig, AnyAgent
from any_agent.tools import search_web, visit_webpage

agent = AnyAgent.create(
    "tinyagent",  # See all options in https://mozilla-ai.github.io/any-agent/
    AgentConfig(
        model_id="mistral:mistral-small-latest", tools=[search_web, visit_webpage]
    ),
)
```

## Run the Agent to Generate a Trace

Now let's run our agent on a test query to generate a trace that we can evaluate.

```python
prompt = """What film won a Goya Award for best film in 2024?
Please provide the name of the film, the genre, a very brief
description of the film - and rotten tomatoes popcornmeter
score."""

agent_trace = agent.run(prompt)
```

## View the Agent Results

Let's first see what our agent produced:

```python
print(f"⏱️ Duration: {agent_trace.duration.total_seconds():.2f}s")
print(f"💰 Cost: ${agent_trace.cost.total_cost:.6f}")

print("\n--- Final Answer ---")
print(agent_trace.final_output)

print("\n--- Tool Execution Path ---")
# Show the exact steps the agent took so we can verify if it "Cheated" or not
for span in agent_trace.spans:
    if span.is_tool_execution():
        print(f"🛠️ Tool Used: {span.attributes.get('gen_ai.tool.name')}")
```

## Method 1: Custom Code Evaluation

Before using LLM-based evaluation, let's start with deterministic custom code evaluation. This is often more efficient, reliable, and cost-effective for specific criteria.

Some criteria are clearly quantitative: a result exists or it doesn't, it has a measurable length, the number of steps can be counted and a tool was either called or wasn't.

```python
from any_agent.tracing.agent_trace import AgentTrace
from any_agent.tracing.attributes import GenAI

def check_tool_usage(trace: AgentTrace, required_tool: str) -> bool:
    """Check if a specific tool was used in the trace."""
    return any(
        span.attributes[GenAI.TOOL_NAME] == required_tool
        for span in trace.spans
        if span.is_tool_execution()
    )

def evaluate_web_search_efficiency(trace: AgentTrace) -> dict:
    """Custom evaluation function for web search agent efficiency criteria."""
    # Direct access to trace properties
    token_count = trace.tokens.total_tokens
    step_count = len(trace.spans)
    final_output = trace.final_output
    duration = trace.duration.total_seconds()
    # Check if web search tools were used
    used_search = check_tool_usage(trace, "search_web")
    used_visit = check_tool_usage(trace, "visit_webpage")
    # Apply quantitative criteria
    results = {
        "token_efficient": token_count
        < 20000,  # Magic number alert: adjust to what you consider reasonable for your budget
        "step_efficient": step_count
        <= 10,  # A high number of steps would point at problems, but this is also a debatable limit
        "has_output": final_output is not None and len(str(final_output)) > 50,
        "used_web_search": used_search,
        "used_webpage_visit": used_visit,
        "reasonable_duration": duration < 60,
    }
    # Choose the quantitative criteria you care most about
    results["passed"] = all(
        [
            results["token_efficient"],
            results["step_efficient"],
            results["has_output"],
            results["used_web_search"],
        ]
    )
    return results
```

```python
evaluation = evaluate_web_search_efficiency(agent_trace)
print("Custom Code Evaluation Results:")
for key, value in evaluation.items():
    print(f"  {key}: {value}")
```

## Method 2: LLM Judge Evaluation

The method above is already useful and can assess quantitative results (how long or how costly answers were, whether a specific tool was present). Programmatic evaluations are less costly, more deterministic, but also less flexible. They can see that a tool was used: but was the result well understood? Was the content actually used to extract an answer? This is a qualitative assessment.

For such criteria, you can use the `LlmJudge`. This is great for evaluating response quality, helpfulness, and other subjective criteria.

### 💡 Good to know: different models

Notice we use a different LLM as a judge to the one we used for the original agent, as LLM judges are known to have a [bias towards their own results](https://neurips.cc/virtual/2024/poster/96672).

```python
from any_agent.evaluation import LlmJudge

# Create an LLM judge
judge = LlmJudge(model_id="mistral:mistral-large-latest")

# Define evaluation questions - notice the last one is not like the others
evaluation_questions = [
    "Did the agent provide a clear and concise answer?",
    "Did the agent correctly identify the genre?",
    "Did the agent include a Rotten Tomatoes score in its response?",
]

# Run evaluations
print("LLM Judge Evaluation Results:")
print("=" * 60)

results = []
for i, question in enumerate(evaluation_questions, 1):
    result = judge.run(context=str(agent_trace.spans_to_messages()), question=question)
    results.append(result)
    print(f"Question {i}: {question}")
    print(f"  Passed: {result.passed}")
    print(f"  Reasoning: {result.reasoning}")
    print("-" * 60)

# Summary
passed_count = sum(1 for r in results if r.passed)
print(f"\nOverall: {passed_count}/{len(results)} criteria passed")
```

### 💡 Good to know: fuzzy criteria

Notice Question 3: if you run the evaluation multiple times, it won't pass or fail consistently, since the LLM judge may interpret that only the description should be under 10 words, not necessarily the whole Agent's answer. In the programmatic method, there is nothing to interpret: we check that the final output was under 10 words.

This showcases the main downside with using an LLM judge: as with humans, criteria can be misunderstood.

On the other hand, using a programmatic approach to assess clarity, for example, would have been rather complex without an LLM judge.

A take-home message here is to use custom code when criteria can be counted or measured, and think of using an LLMJudge when your criteria are qualitative.

## Method 3: Agent Judge Evaluation

For more complex evaluations that require inspecting specific aspects of the trace, we can use the `AgentJudge`. Notice the AgentJudge can:

* call built-in tools to get straight to relevant parts of the traces (e.g. final output),
* call additional tools that the original agent did not have. For example, you will see below how we give it a second search tool so it can do its own research to check if the original agent's answer was correct.

As with the LLMJudge, we choose a different model to the one enabling the original judge.

```python
from any_agent.evaluation import AgentJudge
from any_agent.tools import search_web

# Create an agent judge
agent_judge = AgentJudge(model_id="mistral:mistral-large-latest")

# Define a complex evaluation question that requires trace inspection
complex_question = """
Evaluate the agent's performance on this web search task by verifying
whether the agent correctly used web search to find relevant information
for the winner film of the Goya Award in 2024 and its Rotten Tomatoes rating?

Use the available tools to inspect the trace and, specially, make sure
the agent visited Rotten Tomatoes and checked the audience score, not
the critics score.
"""

# Run the agent judge evaluation
eval_trace = agent_judge.run(
    trace=agent_trace,
    question=complex_question,
    additional_tools=[
        search_web
    ],  # Give the judge access to web search for verification
)

# Get the evaluation result
result = eval_trace.final_output
print("Agent Judge Evaluation Result:")
print("=" * 60)
print(f"Passed: {result.passed}")
print(f"Reasoning: {result.reasoning}")
print("=" * 60)
```

Notice how giving the judge tools enables it to check **independently** whether the original agent successfully did its job.
